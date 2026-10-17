import numpy as np
import pandas as pd

from attp import (
    auto_attp_al_clf,
    auto_attp_al_reg,
    auto_attp_clf,
    auto_attp_reg,
    auto_attp_ts,
    b_attp_clf,
    m_attp_clf,
    m_attp_cluster,
    m_attp_reg,
    m_attp_ts,
    mc_mcls_attp_clf,
    mc_mcls_attp_reg,
    r_d_attp,
    r_f_attp,
    r_ra_attp,
    r_t_attp,
    r_tf_attp,
    s_attp_clf,
    s_attp_cluster,
    s_attp_clustered_sampling,
    s_attp_reg,
    s_attp_ts,
    sattp_split,
)


def _data():
    rng = np.random.default_rng(71)
    labels = pd.Series(np.repeat(["a", "b", "c"], 60), name="class")
    features = pd.DataFrame(
        rng.normal(size=(len(labels), 4)),
        columns=list("abcd"),
        index=pd.Index([f"row-{i}" for i in range(len(labels))]),
    )
    labels.index = features.index
    target = pd.Series(rng.normal(size=len(features)), index=features.index, name="target")
    return features, labels, target


def _assert_supervised_split(result, X, y):
    X_train, X_test, y_train, y_test = result
    assert len(X_train) + len(X_test) == len(X)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(X.index)
    assert y_train.name == y.name
    assert y_test.name == y.name


def test_classifier_variant_splits():
    X, labels, _ = _data()
    _assert_supervised_split(
        m_attp_clf(X, labels, n_clusters=3), X, labels
    )
    _assert_supervised_split(
        mc_mcls_attp_clf(X, labels), X, labels
    )


def test_balanced_split_oversamples_training_only():
    rng = np.random.default_rng(19)
    labels = pd.Series(np.repeat(["major", "minor"], [90, 30]), name="class")
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    result = b_attp_clf(
        X,
        labels,
        max_clusters=2,
        min_clusters=2,
        DR_Comps_Min=1,
        DR_Comps_Max=1,
    )
    X_train, X_test, y_train, y_test = result
    assert y_train.value_counts().nunique() == 1
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(y_train.drop_duplicates())
    assert set(X_test.index).isdisjoint(set(X_train.index))


def test_regression_variant_splits():
    X, _, target = _data()
    for splitter in (
        s_attp_reg,
        m_attp_reg,
        r_t_attp,
        r_f_attp,
        r_tf_attp,
        r_ra_attp,
        r_d_attp,
        mc_mcls_attp_reg,
        auto_attp_reg,
    ):
        _assert_supervised_split(splitter(X, target), X, target)


def test_pool_cluster_and_time_series_variants():
    X, labels, _ = _data()
    for splitter in (auto_attp_al_clf, auto_attp_al_reg):
        train, test = splitter(X)
        assert len(train) + len(test) == len(X)
        assert set(train.index).isdisjoint(test.index)

    for splitter in (s_attp_cluster, m_attp_cluster):
        train, test = splitter(X)
        assert len(train) + len(test) == len(X)
        assert set(train.index).isdisjoint(test.index)

    expected_split = int(len(X) * 0.8)
    for splitter in (s_attp_ts, m_attp_ts, auto_attp_ts):
        train, test, y_train, y_test = splitter(X, labels)
        assert train.index.equals(X.index[:expected_split])
        assert test.index.equals(X.index[expected_split:])
        assert y_train.index.equals(train.index)
        assert y_test.index.equals(test.index)

    train, test = s_attp_ts(X)
    assert train.index.equals(X.index[:expected_split])
    assert test.index.equals(X.index[expected_split:])


def test_classifier_renamed_with_backward_compatible_alias():
    assert sattp_split is s_attp_clf
    X, labels, _ = _data()
    result = s_attp_clf(
        X,
        labels,
        test_size=0.25,
        min_clusters=2,
        max_clusters=2,
        DR_Comps_Min=1,
        DR_Comps_Max=2,
    )
    assert len(result.X_test) > 0
    assert result.best_split == 0.75
    with_reducer = s_attp_clf(
        X,
        labels,
        test_size=0.25,
        min_clusters=2,
        max_clusters=2,
        DR_Comps_Min=1,
        DR_Comps_Max=2,
        dr_reducer=True,
    )
    output, reducer = with_reducer
    assert output.best_split == 0.75
    assert reducer.__class__.__name__ == "LinearDiscriminantAnalysis"


def test_automated_classifier_delegates_to_s_attp_clf():
    X, labels, _ = _data()
    result = auto_attp_clf(
        X,
        labels,
        test_size=0.25,
        min_clusters=2,
        max_clusters=2,
        DR_Comps_Min=1,
        DR_Comps_Max=2,
    )
    assert result.best_split == 0.75


def test_dr_reducer_is_available_for_all_variants():
    X, labels, target = _data()
    methods = (
        (m_attp_clf, (X, labels), {"n_clusters": 3}, "PCA"),
        (
            b_attp_clf,
            (X, labels),
            {"min_clusters": 2, "max_clusters": 2, "DR_Comps_Max": 2},
            "LinearDiscriminantAnalysis",
        ),
        (mc_mcls_attp_clf, (X, labels), {}, None),
        (auto_attp_al_clf, (X,), {}, None),
        (
            auto_attp_clf,
            (X, labels),
            {"min_clusters": 2, "max_clusters": 2, "DR_Comps_Max": 2},
            "LinearDiscriminantAnalysis",
        ),
        (s_attp_reg, (X, target), {"n_clusters": 3}, "PCA"),
        (m_attp_reg, (X, target), {"n_clusters": 3}, "PLSRegression"),
        (r_t_attp, (X, target), {}, None),
        (r_f_attp, (X, target), {"n_clusters": 3}, "PCA"),
        (r_tf_attp, (X, target), {"n_clusters": 3}, "PCA"),
        (r_ra_attp, (X, target), {"n_clusters": 3}, "PCA"),
        (r_d_attp, (X, target), {}, None),
        (mc_mcls_attp_reg, (X, target), {}, None),
        (auto_attp_al_reg, (X,), {}, None),
        (auto_attp_reg, (X, target), {"n_clusters": 3}, "PCA"),
        (s_attp_ts, (X, labels), {}, None),
        (m_attp_ts, (X, labels), {}, None),
        (auto_attp_ts, (X, labels), {}, None),
        (s_attp_cluster, (X,), {"n_clusters": 3}, "PCA"),
        (m_attp_cluster, (X,), {"n_clusters": 3}, None),
    )

    for method, args, kwargs, reducer_name in methods:
        normal_output = method(*args, **kwargs)
        output, reducer = method(*args, dr_reducer=True, **kwargs)
        assert type(output) is type(normal_output), method.__name__
        if reducer_name is None:
            assert reducer is None, method.__name__
        else:
            assert reducer.__class__.__name__ == reducer_name, method.__name__


def test_clustered_sampling_preprocessing_supervised_and_alias():
    assert s_attp_cluster is s_attp_clustered_sampling
    X, labels, _ = _data()
    X = X.copy()
    X["category"] = np.where(np.arange(len(X)) % 2, "odd", "even")
    X["mostly_missing"] = np.nan
    X["event_year"] = np.arange(len(X))
    X.loc[X.index[:8], "a"] = np.nan

    normal_output = s_attp_clustered_sampling(
        X,
        labels,
        test_size=0.25,
        n_clusters=3,
    )
    X_train, X_test, y_train, y_test = normal_output
    assert len(X_train) + len(X_test) == len(X)
    assert "mostly_missing" not in X_train.columns
    assert "event_year" not in X_train.columns
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert X_train["a"].notna().all()
    assert np.issubdtype(X_train["category"].dtype, np.number)


def test_clustered_sampling_unsupervised_custom_reducer_and_candidate_size():
    X, _, _ = _data()
    from sklearn.decomposition import PCA

    output, reducer = s_attp_clustered_sampling(
        X,
        test_size=[0.2, 0.25],
        n_clusters=3,
        dr_reducer=True,
    )
    X_train, X_test = output
    assert len(X_train) + len(X_test) == len(X)
    assert reducer.__class__.__name__ == "PCA"

    custom_reducer = PCA(n_components=2)
    output, fitted_reducer = s_attp_clustered_sampling(
        X,
        reducer_pipeline=custom_reducer,
        n_clusters=3,
        dr_reducer=True,
    )
    assert len(output) == 2
    assert fitted_reducer is custom_reducer
    assert hasattr(fitted_reducer, "components_")


def test_clustered_sampling_outlier_mode_and_singletons():
    X, labels, _ = _data()
    outlier_X = X.copy()
    outlier_X.iloc[-1] = 1000
    inliers, outliers, y_inliers, y_outliers = s_attp_clustered_sampling(
        outlier_X,
        labels,
        o_i=True,
        outlier_quantile=0.05,
    )
    assert len(inliers) + len(outliers) == len(X)
    assert len(outliers) > 0
    assert inliers.index.equals(y_inliers.index)
    assert outliers.index.equals(y_outliers.index)

    unique_features = pd.DataFrame({"value": np.arange(20, dtype=float)})
    unique_target = pd.Series(np.arange(20) % 2)
    X_train, X_test, y_train, y_test = s_attp_clustered_sampling(
        unique_features,
        unique_target,
        n_clusters=20,
        test_size=0.25,
    )
    assert len(X_train) + len(X_test) == len(unique_features)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
