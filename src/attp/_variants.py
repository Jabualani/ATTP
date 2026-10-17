from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ._split import SplitResult, s_attp_clf


def _ensure_pandas(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if isinstance(X, pd.DataFrame):
        features = X.copy()
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame.")
        features = pd.DataFrame(array)
    if features.empty or features.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature.")

    if y is None:
        return features, None
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or a DataFrame with exactly one column.")
        target = y.iloc[:, 0].copy()
    elif isinstance(y, pd.Series):
        target = y.copy()
    else:
        target = pd.Series(y)
    if len(target) != len(features):
        raise ValueError("X and y must contain the same number of rows.")
    target.index = features.index
    if target.isna().any():
        raise ValueError("y must not contain missing values.")
    return features, target


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("X must contain at least one numeric feature.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Numeric features must not contain missing or infinite values.")
    return numeric


def _validate_test_size(test_size: float | int) -> None:
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be an integer or float.")
    if isinstance(test_size, int):
        if test_size < 1:
            raise ValueError("Integer test_size must be at least 1.")
    elif not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")


def _safe_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    strata: pd.Series,
    test_size: float | int = 0.2,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _validate_test_size(test_size)
    if isinstance(test_size, int) and test_size >= len(X):
        raise ValueError("Integer test_size must be smaller than the number of rows.")
    strata = pd.Series(strata.to_numpy(), index=X.index)
    counts = strata.value_counts()
    singleton_positions = np.flatnonzero(strata.map(counts).to_numpy() == 1)
    valid_positions = np.flatnonzero(strata.map(counts).to_numpy() > 1)
    valid_strata = strata.iloc[valid_positions]
    if isinstance(test_size, int) and test_size >= len(valid_positions):
        raise ValueError("Integer test_size must be smaller than eligible row count.")
    n_test = (
        test_size
        if isinstance(test_size, int)
        else int(np.ceil(len(valid_positions) * test_size))
    )
    n_train = len(valid_positions) - n_test
    if len(valid_positions) == 0 or n_test < 1 or n_train < 1:
        raise ValueError("Not enough rows per stratum for a valid train/test split.")
    group_codes, unique_groups = pd.factorize(valid_strata, sort=False)
    group_counts = np.bincount(group_codes)
    max_groups = min(n_train, n_test)
    if len(unique_groups) > max_groups:
        keep_codes = set(np.argsort(group_counts)[::-1][: max_groups - 1])
        other_code = len(unique_groups)
        group_codes = np.array(
            [code if code in keep_codes else other_code for code in group_codes]
        )
    valid_strata = pd.Series(group_codes)

    stratify = valid_strata if valid_strata.nunique() > 1 else None
    train_valid, test_positions = train_test_split(
        valid_positions,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_positions = np.concatenate((train_valid, singleton_positions))
    return (
        X.iloc[train_positions].copy(),
        X.iloc[test_positions].copy(),
        y.iloc[train_positions].copy(),
        y.iloc[test_positions].copy(),
    )


def _create_target_bins(y: pd.Series, n_bins: int) -> pd.Series:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    try:
        values = pd.to_numeric(y, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Regression targets must be numeric.") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Regression targets must not contain missing or infinite values.")
    n_unique = values.nunique()
    if n_unique < 2:
        raise ValueError("Regression targets must contain at least two distinct values.")
    bins = min(n_bins, n_unique)
    return pd.qcut(values, q=bins, labels=False, duplicates="drop").astype(int)


def _cluster_labels(
    Z: np.ndarray,
    n_clusters: int,
    random_state: int | None,
    *,
    method: str = "kmeans",
) -> np.ndarray:
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    count = min(n_clusters, len(Z))
    if method == "gmm":
        clusterer = GaussianMixture(n_components=count, random_state=random_state)
    else:
        clusterer = KMeans(n_clusters=count, random_state=random_state, n_init=10)
    return clusterer.fit_predict(Z)


def _combine_strata(primary: pd.Series, clusters: np.ndarray, index: pd.Index) -> pd.Series:
    cluster_series = pd.Series(clusters, index=index).astype(str)
    return primary.astype(str) + "_" + cluster_series


def _require_target(y: pd.Series | None) -> pd.Series:
    if y is None:
        raise ValueError("y is required for this split method.")
    return y


def _with_reducer(output: Any, reducer: Any, dr_reducer: bool) -> Any:
    return (output, reducer) if dr_reducer else output


def m_attp_clf(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_method: str = "pca",
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Split classification data by target class and PCA/GMM cluster."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    numeric = _numeric_features(X)
    scaled = StandardScaler().fit_transform(numeric)
    reducer = None
    if dr_method.lower() == "pca":
        components = min(5, numeric.shape[1], len(numeric))
        reducer = PCA(n_components=components, random_state=random_state)
        Z = reducer.fit_transform(scaled)
    else:
        Z = scaled
    clusters = _cluster_labels(Z, n_clusters, random_state, method="gmm")
    strata = _combine_strata(target, clusters, X.index)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, reducer, dr_reducer)


def b_attp_clf(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    balance_method: str = "random_over",
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Oversample minority classes in the training portion only."""
    if balance_method != "random_over":
        raise ValueError("balance_method must be 'random_over'.")
    result = s_attp_clf(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        dr_reducer=dr_reducer,
        **kwargs,
    )
    reducer = None
    if dr_reducer:
        result, reducer = result
    if not isinstance(result, SplitResult):
        raise TypeError("s_attp_clf must return a SplitResult.")
    X_train = result.X_train
    y_train = result.y_train
    class_counts = y_train.value_counts()
    max_count = int(class_counts.max())
    rng = np.random.default_rng(random_state)
    positions = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(y_train.to_numpy() == label),
                size=max_count,
                replace=True,
            )
            for label in class_counts.index
        ]
    )
    output = (
        X_train.iloc[positions].copy(),
        result.X_test,
        y_train.iloc[positions].copy(),
        result.y_test,
    )
    return _with_reducer(output, reducer, dr_reducer)


def mc_mcls_attp_clf(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a reproducible class-stratified Monte Carlo split."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    _validate_test_size(test_size)
    output = train_test_split(
        X, target, test_size=test_size, random_state=random_state, stratify=target
    )
    return _with_reducer(output, None, dr_reducer)


def auto_attp_al_clf(
    X_pool: pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Partition an unlabeled pool by K-Means strata."""
    del kwargs
    X_pool, _ = _ensure_pandas(X_pool)
    _validate_test_size(test_size)
    numeric = _numeric_features(X_pool)
    clusters = _cluster_labels(
        StandardScaler().fit_transform(numeric), 2, random_state
    )
    strata = pd.Series(clusters, index=X_pool.index)
    output = _safe_stratified_split(
        X_pool,
        strata,
        strata,
        test_size,
        random_state,
    )[:2]
    return _with_reducer(output, None, dr_reducer)


def auto_attp_clf(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Use the S-ATTP classifier splitter as the automated classification method."""
    kwargs.pop("splits", None)
    return s_attp_clf(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        dr_reducer=dr_reducer,
        **kwargs,
    )


def s_attp_reg(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Stratify regression splits by binned target and PCA/K-Means clusters."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    numeric = _numeric_features(X)
    scaled = StandardScaler().fit_transform(numeric)
    components = min(3, numeric.shape[1], len(numeric))
    reducer = PCA(n_components=components, random_state=random_state)
    Z = reducer.fit_transform(scaled)
    clusters = _cluster_labels(Z, n_clusters, random_state)
    strata = _combine_strata(_create_target_bins(target, n_bins), clusters, X.index)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, reducer, dr_reducer)


def m_attp_reg(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Stratify regression splits by binned target and PLS/GMM clusters."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    numeric = _numeric_features(X)
    scaled = StandardScaler().fit_transform(numeric)
    components = min(3, numeric.shape[1], len(numeric))
    reducer = PLSRegression(n_components=components)
    Z = reducer.fit(scaled, target).transform(scaled)
    clusters = _cluster_labels(Z, n_clusters, random_state, method="gmm")
    strata = _combine_strata(_create_target_bins(target, n_bins), clusters, X.index)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, reducer, dr_reducer)


def r_t_attp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Stratify regression rows by target quantile bins."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    strata = _create_target_bins(target, n_bins)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, None, dr_reducer)


def r_f_attp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Stratify regression rows by feature-space K-Means clusters."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    numeric = _numeric_features(X)
    components = min(3, numeric.shape[1], len(numeric))
    reducer = PCA(n_components=components, random_state=random_state)
    Z = reducer.fit_transform(StandardScaler().fit_transform(numeric))
    clusters = _cluster_labels(Z, n_clusters, random_state)
    strata = pd.Series(clusters, index=X.index)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, reducer, dr_reducer)


def r_tf_attp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Jointly stratify regression rows by binned target and feature clusters."""
    return s_attp_reg(
        X,
        y,
        test_size,
        random_state,
        n_bins,
        n_clusters,
        dr_reducer=dr_reducer,
        **kwargs,
    )


def r_ra_attp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Use target quantile bins as a residual-aware stratification proxy."""
    return s_attp_reg(
        X, y, test_size, random_state, n_bins, dr_reducer=dr_reducer, **kwargs
    )


def r_d_attp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 15,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Stratify regression rows by target quantile bins."""
    return r_t_attp(
        X, y, test_size, random_state, n_bins, dr_reducer=dr_reducer, **kwargs
    )


def mc_mcls_attp_reg(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_bins: int = 10,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a reproducible quantile-stratified regression split."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    target = _require_target(y)
    strata = _create_target_bins(target, n_bins)
    output = _safe_stratified_split(X, target, strata, test_size, random_state)
    return _with_reducer(output, None, dr_reducer)


def auto_attp_al_reg(
    X_pool: pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Partition an unlabeled pool by feature-space cluster."""
    return auto_attp_al_clf(
        X_pool, test_size, random_state, dr_reducer=dr_reducer, **kwargs
    )


def auto_attp_reg(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Use the statistical binned-target regression splitter by default."""
    return s_attp_reg(
        X, y, test_size, random_state, dr_reducer=dr_reducer, **kwargs
    )


def s_attp_ts(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    test_size: float = 0.2,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Split time-series rows in order, without shuffling."""
    del kwargs
    X, y = _ensure_pandas(X, y)
    _validate_test_size(test_size)
    split_idx = int(len(X) * (1.0 - test_size))
    if split_idx == 0 or split_idx == len(X):
        raise ValueError("test_size must leave at least one row in each split.")
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    if y is None:
        output = X_train, X_test
    else:
        output = X_train, X_test, y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    return _with_reducer(output, None, dr_reducer)


def m_attp_ts(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    test_size: float = 0.2,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Apply strict chronological splitting to multivariate time-series data."""
    return s_attp_ts(
        X, y=y, test_size=test_size, dr_reducer=dr_reducer, **kwargs
    )


def auto_attp_ts(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    test_size: float = 0.2,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Automatically partition a time series while preserving chronology."""
    return s_attp_ts(
        X, y=y, test_size=test_size, dr_reducer=dr_reducer, **kwargs
    )


def m_attp_cluster(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int | None = 42,
    n_clusters: int = 5,
    dr_reducer: bool = False,
    **kwargs: Any,
) -> Any:
    """Split feature rows while preserving Gaussian-mixture cluster proportions."""
    del y, kwargs
    X, _ = _ensure_pandas(X)
    numeric = _numeric_features(X)
    clusters = _cluster_labels(
        StandardScaler().fit_transform(numeric), n_clusters, random_state, method="gmm"
    )
    cluster_series = pd.Series(clusters, index=X.index)
    output = _safe_stratified_split(
        X, cluster_series, cluster_series, test_size, random_state
    )[:2]
    return _with_reducer(output, None, dr_reducer)


def s_attp_clustered_sampling(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | np.ndarray | None = None,
    test_size: float | int | list[float | int] = 0.2,
    o_i: bool = False,
    outlier_quantile: float | None = None,
    reducer_pipeline: Any | None = None,
    n_clusters: int | None = None,
    random_state: int | None = 42,
    drop_corr_threshold: float = 0.95,
    max_missing_rate: float = 0.30,
    row_threshold_minibatch: int = 50_000,
    dr_reducer: bool = False,
) -> Any:
    """Split rows using supervised or unsupervised reduced-space clusters.

    If ``test_size`` is a list, candidates are ranked by how many observed
    strata can fit in both partitions; ties preserve the input order.
    """
    if isinstance(X, pd.DataFrame):
        features = X.copy()
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame.")
        features = pd.DataFrame(array)
    if features.empty:
        raise ValueError("X must contain at least one row and one feature.")

    target: pd.Series | None = None
    if y is not None:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 0:
                raise ValueError("y DataFrame must contain at least one column.")
            target = y.iloc[:, 0].copy()
            labeled_index = True
        elif isinstance(y, pd.Series):
            target = y.copy()
            labeled_index = True
        else:
            target = pd.Series(y)
            labeled_index = False
        if labeled_index:
            common_index = features.index.intersection(target.index)
            if common_index.empty:
                if len(target) != len(features):
                    raise ValueError("X and y must share at least one index or have equal lengths.")
                target.index = features.index
            else:
                features = features.loc[common_index]
                target = target.loc[common_index]
        else:
            if len(target) != len(features):
                raise ValueError("X and y must contain the same number of rows.")
            target.index = features.index
        if target.isna().any():
            raise ValueError("y must not contain missing values.")
        if features.empty:
            raise ValueError("X and y have no aligned rows.")

    if not 0.0 <= max_missing_rate <= 1.0:
        raise ValueError("max_missing_rate must be between 0 and 1.")
    if not 0.0 <= drop_corr_threshold <= 1.0:
        raise ValueError("drop_corr_threshold must be between 0 and 1.")
    if row_threshold_minibatch < 1:
        raise ValueError("row_threshold_minibatch must be at least 1.")

    reducer: Any | None = reducer_pipeline
    if reducer_pipeline is not None:
        if hasattr(reducer_pipeline, "fit_transform"):
            fit_transform = reducer_pipeline.fit_transform
            steps = getattr(reducer_pipeline, "steps", ())
            final_estimator = steps[-1][1] if steps else reducer_pipeline
            if target is not None and isinstance(
                final_estimator, LinearDiscriminantAnalysis
            ):
                latent = fit_transform(features, target)
            else:
                latent = fit_transform(features)
        elif hasattr(reducer_pipeline, "transform"):
            latent = reducer_pipeline.transform(features)
        else:
            raise TypeError("reducer_pipeline must provide fit_transform() or transform().")
        clean_features = features.copy()
    else:
        clean_features = features.copy()
        date_columns = [
            column
            for column in clean_features.columns
            if any(
                term in str(column).lower()
                for term in ("date", "timestamp", "time", "year")
            )
        ]
        clean_features = clean_features.drop(columns=date_columns, errors="ignore")
        missing_rates = clean_features.isna().mean()
        clean_features = clean_features.drop(
            columns=missing_rates[missing_rates > max_missing_rate].index
        )
        if clean_features.shape[1] == 0:
            raise ValueError("Preprocessing removed every feature column.")

        numeric_columns = clean_features.select_dtypes(include=[np.number]).columns
        categorical_columns = clean_features.columns.difference(numeric_columns)
        if len(numeric_columns):
            numeric_values = clean_features[numeric_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            numeric_values = SimpleImputer(strategy="median").fit_transform(
                numeric_values
            )
            for column_position, column in enumerate(numeric_columns):
                clean_features[column] = numeric_values[:, column_position]
        if len(categorical_columns):
            categorical_values = SimpleImputer(
                strategy="most_frequent"
            ).fit_transform(clean_features[categorical_columns])
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            categorical_values = encoder.fit_transform(categorical_values)
            for column_position, column in enumerate(categorical_columns):
                clean_features[column] = categorical_values[:, column_position]

        if clean_features.shape[1] > 1:
            correlation = clean_features.corr(numeric_only=True).abs()
            upper = correlation.where(
                np.triu(np.ones(correlation.shape), k=1).astype(bool)
            )
            correlated_columns = [
                column
                for column in upper.columns
                if (upper[column] > drop_corr_threshold).any()
            ]
            clean_features = clean_features.drop(
                columns=correlated_columns, errors="ignore"
            )
        if clean_features.shape[1] == 0:
            raise ValueError("Preprocessing removed every feature column.")

        scaled_features = StandardScaler().fit_transform(clean_features)
        if target is not None:
            n_components = min(clean_features.shape[1], target.nunique() - 1)
            if n_components >= 1:
                reducer = LinearDiscriminantAnalysis(n_components=n_components)
                latent = reducer.fit_transform(scaled_features, target)
            else:
                reducer = None
                latent = scaled_features
        else:
            reducer = PCA(
                n_components=min(5, clean_features.shape[1], len(clean_features)),
                random_state=random_state,
            )
            latent = reducer.fit_transform(scaled_features)

    latent = np.asarray(latent)
    if latent.ndim == 1:
        latent = latent.reshape(-1, 1)
    if latent.ndim != 2 or latent.shape[0] != len(clean_features):
        raise ValueError("Reducer output must be a 2D array with one row per input row.")
    if not np.isfinite(latent).all():
        raise ValueError("Reduced features must not contain missing or infinite values.")

    if o_i:
        if outlier_quantile is not None and not 0.0 < outlier_quantile < 1.0:
            raise ValueError("outlier_quantile must be strictly between 0 and 1.")
        centroid = np.mean(latent, axis=0)
        distances = np.linalg.norm(latent - centroid, axis=1)
        if outlier_quantile is not None:
            cutoff = np.quantile(distances, 1.0 - outlier_quantile)
        else:
            if len(distances) < 2:
                raise ValueError("At least two rows are required for automatic outlier detection.")
            ordered_distances = np.sort(distances)
            gaps = np.diff(ordered_distances)
            tail_start = min(int(len(distances) * 0.70), len(gaps) - 1)
            max_gap = tail_start + int(np.argmax(gaps[tail_start:]))
            cutoff = ordered_distances[max_gap + 1]
        outlier_mask = distances >= cutoff
        inlier_X = clean_features.iloc[np.flatnonzero(~outlier_mask)].copy()
        outlier_X = clean_features.iloc[np.flatnonzero(outlier_mask)].copy()
        if target is None:
            output = (inlier_X, outlier_X)
        else:
            inliers = np.flatnonzero(~outlier_mask)
            outliers = np.flatnonzero(outlier_mask)
            output = (
                inlier_X,
                outlier_X,
                target.iloc[inliers].copy(),
                target.iloc[outliers].copy(),
            )
        return _with_reducer(output, reducer, dr_reducer)

    if isinstance(test_size, list):
        candidates = test_size
    else:
        candidates = [test_size]
    if not candidates:
        raise ValueError("test_size must contain at least one candidate.")
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise TypeError("test_size candidates must be integers or floats.")
        if isinstance(candidate, int):
            if not 0 < candidate < len(clean_features):
                raise ValueError("Integer test_size must be between 1 and n_samples - 1.")
        elif not 0.0 < candidate < 1.0:
            raise ValueError("Float test_size candidates must be strictly between 0 and 1.")

    if n_clusters is None:
        cluster_count = min(10, max(2, len(clean_features) // 100))
    else:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        cluster_count = min(n_clusters, len(clean_features))
    if len(clean_features) <= row_threshold_minibatch:
        cluster_model = KMeans(
            n_clusters=cluster_count, random_state=random_state, n_init=10
        )
    else:
        cluster_model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            batch_size=2048,
            n_init=3,
        )
    cluster_labels = cluster_model.fit_predict(latent)
    if target is None:
        strata = pd.Series(cluster_labels, index=clean_features.index)
    else:
        strata = _combine_strata(target, cluster_labels, clean_features.index)

    if len(candidates) > 1:
        counts = strata.value_counts()
        number_of_strata = len(counts)

        def candidate_score(candidate: float | int) -> int:
            if isinstance(candidate, int):
                test_rows = candidate
            else:
                test_rows = int(np.ceil(len(clean_features) * candidate))
            train_rows = len(clean_features) - test_rows
            return min(number_of_strata, test_rows, train_rows)

        chosen_size = max(candidates, key=candidate_score)
    else:
        chosen_size = candidates[0]

    stratum_counts = strata.value_counts()
    eligible_row_count = int(strata.map(stratum_counts).gt(1).sum())
    if not (stratum_counts >= 2).any() or (
        isinstance(chosen_size, int) and chosen_size >= eligible_row_count
    ):
        if target is None:
            X_train, X_test = train_test_split(
                clean_features, test_size=chosen_size, random_state=random_state
            )
            output = (X_train, X_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                clean_features,
                target,
                test_size=chosen_size,
                random_state=random_state,
            )
            output = (X_train, X_test, y_train, y_test)
    else:
        output = _safe_stratified_split(
            clean_features,
            strata,
            strata,
            test_size=chosen_size,
            random_state=random_state,
        )
        X_train, X_test, _, _ = output

    if target is not None:
        X_train, X_test, _, _ = output
        y_train = target.loc[X_train.index]
        y_test = target.loc[X_test.index]
        output = (X_train, X_test, y_train, y_test)
    else:
        output = (X_train, X_test)
    return _with_reducer(output, reducer, dr_reducer)


s_attp_cluster = s_attp_clustered_sampling
