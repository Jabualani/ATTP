from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

ScoringFunction = Callable[[Any, pd.DataFrame, pd.Series], float]


class NoValidSplitError(ValueError):
    """Raised when no candidate split can be evaluated."""


@dataclass(frozen=True)
class SplitResult:
    """The selected train/test data and scores for an S-ATTP split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    best_split: float
    best_n_clusters: int
    n_lda_components: int
    train_score: float
    test_score: float
    metrics: pd.DataFrame
    train_indices: pd.Index
    test_indices: pd.Index

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        yield self.X_train
        yield self.y_train
        yield self.X_test
        yield self.y_test

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int | slice) -> Any:
        return (self.X_train, self.y_train, self.X_test, self.y_test)[index]


def _as_features(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        features = X.copy()
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame.")
        features = pd.DataFrame(array)

    if features.empty or features.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature.")
    try:
        values = features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain only numeric feature values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("X must not contain missing or infinite values.")
    return features


def _as_target(
    y: pd.Series | pd.DataFrame | Sequence[Any] | np.ndarray,
    index: pd.Index,
) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or a DataFrame with exactly one column.")
        target = y.iloc[:, 0].copy()
    elif isinstance(y, pd.Series):
        target = y.copy()
    else:
        target = pd.Series(y)

    if len(target) != len(index):
        raise ValueError("X and y must contain the same number of rows.")
    target.index = index
    if target.isna().any():
        raise ValueError("y must not contain missing values.")
    if target.nunique() < 2:
        raise ValueError("y must contain at least two classes.")
    return target


def _fit_lda(
    X: pd.DataFrame,
    y: pd.Series,
    min_components: int,
    max_components: int,
    variance_threshold: float,
) -> tuple[LinearDiscriminantAnalysis, int]:
    max_possible = min(X.shape[1], y.nunique() - 1)
    if max_possible < 1:
        raise ValueError("LDA requires at least two target classes and one feature.")

    first = max(1, min(min_components, max_possible))
    last = min(max_components, max_possible)
    if first > last:
        first = last

    selected: LinearDiscriminantAnalysis | None = None
    selected_components = 0
    for n_components in range(first, last + 1):
        candidate = LinearDiscriminantAnalysis(n_components=n_components)
        candidate.fit(X, y)
        selected = candidate
        selected_components = n_components
        if candidate.explained_variance_ratio_.sum() >= variance_threshold:
            break

    if selected is None:
        raise ValueError("No valid LDA component count is available.")
    return selected, selected_components


def sattp_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | pd.DataFrame | Sequence[Any] | np.ndarray,
    *,
    splits: Sequence[float] = (0.7, 0.8),
    min_clusters: int = 2,
    max_clusters: int = 10,
    DR_Comps_Min: int = 1,
    DR_Comps_Max: int = 3,
    model: ClassifierMixin | None = None,
    scoring: ScoringFunction | None = None,
    variance_threshold: float = 0.85,
    random_state: int | None = 0,
    verbose: bool = False,
) -> SplitResult:
    """Find a high-scoring cluster-stratified train/test split.

    LDA and clustering are fitted on the full input before splitting, so the
    returned score is useful for selecting a representative split, not for
    unbiased model evaluation.
    """
    features = _as_features(X)
    target = _as_target(y, features.index)

    if len(splits) == 0:
        raise ValueError("splits must contain at least one train fraction.")
    if any(not 0.0 < split < 1.0 for split in splits):
        raise ValueError("Each split must be a train fraction strictly between 0 and 1.")
    if min_clusters < 1 or max_clusters < min_clusters:
        raise ValueError("Cluster bounds must satisfy 1 <= min_clusters <= max_clusters.")
    if max_clusters > len(features):
        raise ValueError("max_clusters cannot exceed the number of input rows.")
    if DR_Comps_Min < 1 or DR_Comps_Max < DR_Comps_Min:
        raise ValueError(
            "LDA component bounds must satisfy "
            "1 <= DR_Comps_Min <= DR_Comps_Max."
        )
    if not 0.0 <= variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be between 0 and 1.")

    if verbose:
        print(
            f"Fitting LDA components ({DR_Comps_Min}-{DR_Comps_Max})...",
            flush=True,
        )
    lda, n_components = _fit_lda(
        features,
        target,
        DR_Comps_Min,
        DR_Comps_Max,
        variance_threshold,
    )
    if verbose:
        print(
            f"LDA complete: {n_components} component(s); "
            f"evaluating {max_clusters - min_clusters + 1} cluster counts "
            f"across {len(splits)} train fraction(s).",
            flush=True,
        )
    reduced = lda.transform(features)
    candidate_model = model if model is not None else LinearDiscriminantAnalysis()
    n_samples = len(features)
    use_minibatch = n_samples > 50_000
    records: list[dict[str, Any]] = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        if verbose:
            print(f"Fitting clusters: k={n_clusters}...", flush=True)
        if use_minibatch:
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=4096,
                n_init="auto",
            )
        else:
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto",
            )
        cluster_labels = clusterer.fit_predict(reduced)
        strata = pd.Series(
            target.astype(str).to_numpy() + "\x1f" + cluster_labels.astype(str)
        )
        counts = strata.value_counts()
        singleton_positions = np.flatnonzero(strata.map(counts).to_numpy() == 1)
        valid_positions = np.flatnonzero(strata.map(counts).to_numpy() > 1)
        valid_strata = strata.iloc[valid_positions]
        n_valid = len(valid_positions)
        n_groups = valid_strata.nunique()

        for split in splits:
            n_train = int(np.floor(n_valid * split))
            n_test = n_valid - n_train
            if n_groups == 0 or n_train < n_groups or n_test < n_groups:
                if verbose:
                    print(
                        f"  Skipping split={split:.3f}, k={n_clusters}: "
                        "not enough rows per stratum.",
                        flush=True,
                    )
                continue

            train_valid, test_positions = train_test_split(
                valid_positions,
                train_size=split,
                random_state=random_state,
                stratify=valid_strata,
            )
            train_positions = np.concatenate((train_valid, singleton_positions))
            X_train = features.iloc[train_positions]
            y_train = target.iloc[train_positions]
            X_test = features.iloc[test_positions]
            y_test = target.iloc[test_positions]

            fitted_model = clone(candidate_model)
            fitted_model.fit(X_train, y_train)
            if scoring is None:
                train_score = float(fitted_model.score(X_train, y_train))
                test_score = float(fitted_model.score(X_test, y_test))
            else:
                train_score = float(scoring(fitted_model, X_train, y_train))
                test_score = float(scoring(fitted_model, X_test, y_test))
            if not np.isfinite(train_score) or not np.isfinite(test_score):
                raise ValueError("The scoring function must return finite numeric scores.")

            records.append(
                {
                    "Split": split,
                    "n_clusters": n_clusters,
                    "train_score": round(train_score, 3),
                    "test_score": round(test_score, 3),
                    "train_indices": features.index.take(train_positions),
                    "test_indices": features.index.take(test_positions),
                    "_train_positions": train_positions,
                    "_test_positions": test_positions,
                }
            )
            if verbose:
                print(
                    f"  Evaluated split={split:.3f}, k={n_clusters}: "
                    f"train={train_score:.3f}, test={test_score:.3f}.",
                    flush=True,
                )

    if not records:
        if verbose:
            print("No valid candidate split was found.", flush=True)
        raise NoValidSplitError(
            "No valid candidate split was found. Try different train fractions or cluster bounds."
        )

    metrics = pd.DataFrame.from_records(records).sort_values(
        ["test_score", "n_clusters"], ascending=[False, False], ignore_index=True
    )
    best = metrics.iloc[0]
    train_indices = best["train_indices"]
    test_indices = best["test_indices"]
    train_positions = best["_train_positions"]
    test_positions = best["_test_positions"]
    metrics = metrics.drop(columns=["_train_positions", "_test_positions"])

    result = SplitResult(
        X_train=features.iloc[train_positions].copy(),
        y_train=target.iloc[train_positions].copy(),
        X_test=features.iloc[test_positions].copy(),
        y_test=target.iloc[test_positions].copy(),
        best_split=float(best["Split"]),
        best_n_clusters=int(best["n_clusters"]),
        n_lda_components=n_components,
        train_score=float(best["train_score"]),
        test_score=float(best["test_score"]),
        metrics=metrics,
        train_indices=train_indices.copy(),
        test_indices=test_indices.copy(),
    )
    if verbose:
        print(
            "Best split: "
            f"train fraction={result.best_split:.3f}, "
            f"clusters={result.best_n_clusters}, "
            f"test score={result.test_score:.3f}.",
            flush=True,
        )
    return result


def S_ATTP(
    Input_X_df: pd.DataFrame,
    Input_y_df: pd.DataFrame,
    Target_Var_: str,
    Splits: Sequence[float],
    Min_clusters: int,
    Max_Clusters: int,
    DR_Comps_Min: int,
    DR_Comps_Max: int,
    accuracy_model: ClassifierMixin | None,
    return_type: str,
    return_split_metadata: bool,
    verbose: bool,
) -> tuple[Any, ...]:
    """Compatibility wrapper for the original S_ATTP function signature."""
    if Target_Var_ not in Input_y_df.columns:
        raise ValueError(f"Target_Var_ {Target_Var_!r} is not a column in Input_y_df.")
    if return_type not in {"frames", "indices"}:
        raise ValueError("return_type must be either 'frames' or 'indices'.")

    try:
        result = sattp_split(
            Input_X_df,
            Input_y_df[Target_Var_],
            splits=Splits,
            min_clusters=Min_clusters,
            max_clusters=Max_Clusters,
            DR_Comps_Min=DR_Comps_Min,
            DR_Comps_Max=DR_Comps_Max,
            model=accuracy_model,
            verbose=verbose,
        )
    except NoValidSplitError:
        output = (None, None) if return_type == "indices" else (None, None, None, None)
        return output + (None,) if return_split_metadata else output

    if return_type == "indices":
        output = (result.train_indices, result.test_indices)
    else:
        output = (result.X_train, result.y_train, result.X_test, result.y_test)
    return output + (result.metrics,) if return_split_metadata else output
