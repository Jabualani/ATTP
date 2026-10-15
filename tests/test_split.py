import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from attp import S_ATTP, NoValidSplitError, sattp_split


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(12)
    labels = np.repeat(["a", "b", "c"], 40)
    features = rng.normal(size=(len(labels), 5)) + np.repeat(
        np.array([[-2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [2, 0, 0, 0, 0]]),
        40,
        axis=0,
    )
    index = pd.Index([f"sample-{i}" for i in range(len(labels))])
    return pd.DataFrame(features, columns=list("abcde"), index=index), pd.Series(
        labels, index=index, name="label"
    )


def test_split_result_preserves_indices_and_unpacks(classification_data):
    X, y = classification_data
    result = sattp_split(
        X,
        y,
        splits=[0.7],
        min_clusters=2,
        max_clusters=2,
        DR_Comps_Min=1,
        DR_Comps_Max=3,
        model=LogisticRegression(max_iter=500),
        random_state=4,
    )
    X_train, y_train, X_test, y_test = result

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert set(result.train_indices) | set(result.test_indices) == set(X.index)
    assert result.best_split == 0.7
    assert len(result.metrics) == 1
    assert len(result.explained_variance_ratio) == result.n_lda_components
    assert result.total_explained_variance == pytest.approx(
        sum(result.explained_variance_ratio)
    )
    assert 0.0 < result.total_explained_variance <= 1.0


def test_clusters_are_fitted_once_for_each_cluster_count(classification_data, monkeypatch):
    X, y = classification_data
    fitted_cluster_counts = []
    original_fit_predict = KMeans.fit_predict

    def count_fit_predict(self, *args, **kwargs):
        fitted_cluster_counts.append(self.n_clusters)
        return original_fit_predict(self, *args, **kwargs)

    monkeypatch.setattr(KMeans, "fit_predict", count_fit_predict)
    result = sattp_split(
        X,
        y,
        splits=[0.65, 0.75],
        min_clusters=2,
        max_clusters=3,
    )

    assert fitted_cluster_counts == [2, 3]
    assert len(result.metrics) == 4


def test_verbose_prints_progress_and_final_summary(classification_data, capsys):
    X, y = classification_data

    sattp_split(
        X,
        y,
        splits=[0.7],
        min_clusters=2,
        max_clusters=2,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "Fitting LDA components" in output
    assert "LDA complete:" in output
    assert "Fitting clusters: k=2" in output
    assert "Evaluated split=0.700, k=2:" in output
    assert "Best split:" in output


def test_verbose_false_does_not_print(classification_data, capsys):
    X, y = classification_data

    sattp_split(X, y, splits=[0.7], min_clusters=2, max_clusters=2)

    assert capsys.readouterr().out == ""


def test_legacy_wrapper_returns_frames_indices_and_metadata(classification_data):
    X, y = classification_data
    target = y.to_frame()
    target.index = target.index[::-1]
    target = target.reset_index(drop=True)
    X = X.reset_index(drop=True)

    frames = S_ATTP(X, target, "label", [0.7], 2, 2, 1, 2, None, "frames", False, False)
    indices = S_ATTP(X, target, "label", [0.7], 2, 2, 1, 2, None, "indices", True, False)

    assert len(frames) == 4
    assert len(indices) == 3
    assert isinstance(indices[2], pd.DataFrame)


@pytest.mark.parametrize(
    ("X", "y", "kwargs", "message"),
    [
        ([1, 2], [0, 1], {}, "two-dimensional"),
        ([[1, np.nan], [2, 3]], [0, 1], {}, "missing or infinite"),
        ([[1], [2]], [0], {}, "same number of rows"),
        ([[1], [2]], [1, 1], {}, "at least two classes"),
        ([[1], [2]], [0, 1], {"splits": [1.0]}, "strictly between"),
    ],
)
def test_invalid_input_raises_clear_error(X, y, kwargs, message):
    with pytest.raises(ValueError, match=message):
        sattp_split(X, y, **kwargs)


def test_no_candidate_split_raises_specific_error(classification_data):
    X, y = classification_data
    with pytest.raises(NoValidSplitError, match="No valid candidate"):
        sattp_split(X, y, splits=[0.99], min_clusters=2, max_clusters=2)
