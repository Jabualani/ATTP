# attp (Adaptive Train Test Partitioner)

Cluster-stratified train/test splitting on an LDA-reduced feature space (S-ATTP).

Instead of stratifying on the class label alone, `sattp` also stratifies on *where a row sits in
feature space*, so the test set covers the same sub-populations as the train set.

## How it works

1. Reduce the features with **LDA**, using the fewest components that reach 85% of the explained
   variance (configurable).
2. For each candidate cluster count, run **K-Means** once on the LDA components and reuse those
   clusters across candidate train fractions (`MiniBatchKMeans` above 50,000 rows).
3. Build strata from `class × cluster` and do a stratified split. Strata with a single member
   can't be split, so those rows always go to the train set.
4. Score a classifier on every `(train fraction, n_clusters)` pair and return the split with the
   best test score.

## Process Flow Diagram
![process flow diagram](https://raw.githubusercontent.com/Jabualani/ATTP/main/image.png)

## Install

```bash
pip install i-attp
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/Jabualani/ATTP.git
```

For development:

```bash
git clone https://github.com/Jabualani/ATTP.git
cd ATTP
pip install -e ".[dev]"
pytest
```

## Version history and migration

| Version | Main functionality |
| --- | --- |
| `0.1.2` | The original S-ATTP classifier split, exposed as `sattp_split`, plus the legacy `S_ATTP` wrapper. |
| `0.2.0` | The classifier's preferred name is `s_attp_clf`; `sattp_split` and `S_ATTP` remain available. Adds classification, regression, time-series, and clustering variants, including `s_attp_clustered_sampling`. |
| `0.2.1` | Includes the ATTP Framework Reference Architecture PDF in the repository and package distributions. |
| `0.2.2` | Expands the README with parameter references and examples for classification and clustered sampling. |
| `0.2.3` | Uses absolute GitHub URLs for the README image and framework PDF so both resolve in the PyPI description. |

The `0.2.0` release also adds the optional `dr_reducer=True` output for retrieving a fitted
dimensionality reducer. The original `s_attp_clf` call continues to return its `SplitResult` by
default; most added variants return `(X_train, X_test, y_train, y_test)`, while time-series,
pool, clustering, and outlier/inlier methods have their own output shapes as documented below.

To migrate from `0.1.2`, prefer the new import name:

```python
from attp import s_attp_clf

result = s_attp_clf(X, y)
```

Existing code can continue importing `sattp_split` or `S_ATTP`. Documentation-only changes do not
inherently require a version bump; this update is designated as `0.2.3` and synchronizes the
package metadata. For future releases, reserve patch increments for bug fixes and documentation
updates, and use a minor increment for backward-compatible API additions while the project remains
in the `0.x` series. Review return shapes and behavior before each release.

## Classification usage

### `s_attp_clf` parameters

`s_attp_clf(X, y, ...)` evaluates candidate train fractions and cluster counts, then returns the
split with the highest classifier score. `X` must be a numeric pandas `DataFrame` or 2D NumPy
array; `y` may be a pandas `Series`/`DataFrame` or a 1D sequence/array of class labels. Rows must
align, and features must not contain missing or non-numeric values.

| Parameter | Accepted type; default | Meaning and when to use it |
| --- | --- | --- |
| `X` | `DataFrame` or 2D `ndarray`; required | Feature matrix. DataFrame row labels are preserved in the result. |
| `y` | `Series`, one-column `DataFrame`, 1D sequence or `ndarray`; required | Classification target; missing labels are not accepted. |
| `splits` | Sequence of `float`; `(0.7, 0.8)` | Candidate **train fractions**, each strictly between 0 and 1. Use several values to compare train/test proportions. |
| `min_clusters` | `int`; `2` | Smallest K-Means cluster count to evaluate. Must be at least 1. |
| `max_clusters` | `int`; `10` | Largest cluster count to evaluate; cannot exceed the number of rows. Larger ranges take longer. |
| `DR_Comps_Min` | `int`; `1` | Minimum number of LDA components considered. |
| `DR_Comps_Max` | `int`; `3` | Maximum number of LDA components considered; must be at least `DR_Comps_Min`. |
| `model` | scikit-learn `ClassifierMixin` or `None`; `None` | Classifier used to score candidates. `None` uses scikit-learn LDA. Supply a compatible estimator to optimize for a different classifier. |
| `scoring` | callable or `None`; `None` | Optional function `(fitted_model, X, y) -> number`. With `None`, the estimator's `.score()` is used. Scores must be finite. |
| `variance_threshold` | `float` from `0` to `1`; `0.85` | Target cumulative explained-variance ratio used when choosing LDA component count. |
| `random_state` | `int` or `None`; `0` | Seed for repeatable clustering, splitting, and model evaluation. |
| `verbose` | `bool`; `False` | Print progress, skipped candidates, scores, and the selected split. |
| `test_size` | `float` or `None`; `None` | Optional single test fraction. If supplied, it overrides `splits` with train fraction `1 - test_size`. |
| `dr_reducer` | `bool`; `False` | If `True`, return `(SplitResult, fitted_lda)` instead of only `SplitResult`; useful for inspecting or reusing the fitted feature reducer. |

The result unpacks in this order: `(X_train, y_train, X_test, y_test)`. It also provides
`best_split`, `best_n_clusters`, `n_lda_components`, explained-variance information, candidate
`metrics`, and train/test indices. Rows in singleton class-cluster strata are kept in training.
The scores are for selecting a representative split, not unbiased model evaluation (see
[Caveats](#caveats)).

### Classification examples

```python
from attp import s_attp_clf

result = s_attp_clf(
    X, y,
    splits=[0.7, 0.75, 0.8],   # candidate train fractions
    min_clusters=2,
    max_clusters=10,
)

X_train, y_train, X_test, y_test = result      # unpacks like a 4-tuple
print(result.best_split, result.best_n_clusters, result.n_lda_components)
print(result.explained_variance_ratio)          # ratio explained by each LDA component
print(result.total_explained_variance)           # combined ratio for the selected components
print(result.metrics.head())                    # every candidate, best first
```

Row labels of the original `X`/`y` are preserved in the outputs.

`s_attp_clf` is the current name for the original S-ATTP split function. The previous
`sattp_split` name remains available as a backward-compatible alias.

Useful options: `model=` (any scikit-learn classifier used for scoring; defaults to LDA),
`scoring=` (custom `(fitted_model, X, y) -> float`), `variance_threshold=`, `random_state=`,
`test_size=` to provide one holdout size (otherwise `splits` contains candidate train fractions),
and `verbose=True` to print progress and the selected split. Set `DR_Comps_Min=` and
`DR_Comps_Max=` to choose the LDA component range.

Set `dr_reducer=True` on any splitter to receive `(normal_output, reducer)`. The reducer is the
fitted LDA, PCA, or PLS estimator where that method uses one; splitters that do not reduce features
return `None` as the reducer. The default `dr_reducer=False` preserves the normal return value.

### Other classification functions

Classification variants accept `X` as a pandas `DataFrame` or 2D NumPy array, and labeled
variants accept `y` as a pandas `Series`/`DataFrame` or 1D array/sequence. Their usual split
output is `(X_train, X_test, y_train, y_test)`; pool splitting returns `(X_train, X_test)`.
`dr_reducer=True` wraps that output as `(normal_output, reducer)`; methods without a reducer
provide `None`.

| Function | Parameters beyond `X` / `y` | Notes |
| --- | --- | --- |
| `m_attp_clf` | `test_size: float=0.2`, `random_state: Optional[int]=42`, `dr_method: str="pca"`, `n_clusters: int=5`, `dr_reducer: bool=False`, `**kwargs` | PCA is applied when `dr_method="pca"`; other values use standardized features directly. A Gaussian mixture creates clusters. `**kwargs` is accepted but ignored. |
| `b_attp_clf` | `test_size: float=0.2`, `random_state: Optional[int]=42`, `balance_method: str="random_over"`, `dr_reducer: bool=False`, `**kwargs` | Only `"random_over"` is supported. Oversampling affects training rows only. `**kwargs` is forwarded to `s_attp_clf`. |
| `mc_mcls_attp_clf` | `test_size: float=0.2`, `random_state: Optional[int]=42`, `dr_reducer: bool=False`, `**kwargs` | Reproducible class-stratified split; `**kwargs` is accepted but ignored. |
| `auto_attp_al_clf` | `test_size: float=0.2`, `random_state: Optional[int]=42`, `dr_reducer: bool=False`, `**kwargs` | Splits an unlabeled `X_pool` into two feature sets using K-Means; `**kwargs` is accepted but ignored. |
| `auto_attp_clf` | `test_size: float=0.2`, `random_state: Optional[int]=42`, `dr_reducer: bool=False`, `**kwargs` | Delegates to `s_attp_clf`; forwards `**kwargs` except `splits`, which is discarded in favor of `test_size`. |

Other variants return scikit-learn-ordered tuples: `(X_train, X_test, y_train, y_test)`;
pool and clustering functions without targets return `(X_train, X_test)`.

```python
from attp import (
    b_attp_clf, m_attp_clf, mc_mcls_attp_clf,
    s_attp_reg, m_attp_reg, r_t_attp, r_f_attp,
    s_attp_ts,
)

# PCA/GMM classifier split; random oversampling is applied to training rows only.
X_train, X_test, y_train, y_test = m_attp_clf(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = b_attp_clf(X, y, test_size=0.2)

# Regression: target-bin and feature-aware options.
X_train, X_test, y_train, y_test = s_attp_reg(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = m_attp_reg(X, y, test_size=0.2)
r_t_attp(X, y)  # target-bin stratification
r_f_attp(X, y)  # feature-cluster stratification

# Temporal splits preserve row order; clustered sampling accepts labeled or unlabeled data.
X_train, X_test, y_train, y_test = s_attp_ts(X, y)
```

## Clustering usage

### `s_attp_clustered_sampling` parameters

This method preprocesses mixed feature data, reduces it with supervised LDA (when `y` is supplied)
or unsupervised PCA (when `y=None`), then partitions rows by cluster strata. `X` accepts a pandas
`DataFrame` or 2D NumPy array; optional `y` accepts a pandas `Series`/`DataFrame` or 1D array.
For labeled data, shared DataFrame/Series indices are aligned; array targets are matched by row
position. By default it imputes missing values, encodes categorical columns, removes date-like and
high-missingness columns, and filters highly correlated columns.

| Parameter | Accepted type; default | Meaning and when to use it |
| --- | --- | --- |
| `X` | `DataFrame` or 2D `ndarray`; required | Feature matrix. It must contain at least two rows and one feature for clustering. |
| `y` | `Series`, `DataFrame`, 1D `ndarray` or `None`; `None` | Optional labels. With labels, reduction and strata use target class; without them, PCA and cluster-only strata are used. |
| `test_size` | `float`, `int`, or list of floats/ints; `0.2` | Test fraction (`0 < value < 1`) or absolute test-row count (`1` to `n_rows-1`). A list supplies candidates; the method chooses the candidate with the best row/strata feasibility score (ties keep the first). |
| `o_i` | `bool`; `False` | Enable outlier/inlier mode instead of train/test splitting. Returns inlier rows before outlier rows, followed by corresponding targets when `y` is supplied. |
| `outlier_quantile` | `float` or `None`; `None` | In `o_i` mode, choose an upper-tail distance cutoff; e.g. `0.05` uses the 95th-percentile cutoff. Must be strictly between 0 and 1. With `None`, the method estimates a cutoff from the largest gap in the distance tail. |
| `reducer_pipeline` | scikit-learn-style transformer/pipeline or `None`; `None` | Optional custom reducer. It is fit with `fit_transform` when available (LDA receives `y`); otherwise `transform` is used. Custom preprocessing is responsible for handling raw data. |
| `n_clusters` | `int` or `None`; `None` | Requested number of clusters; explicit values are capped at the row count. `None` uses `min(10, max(2, n_rows // 100))`; smaller datasets therefore default to 2 clusters. |
| `random_state` | `int` or `None`; `42` | Seed for PCA/K-Means and splitting. |
| `drop_corr_threshold` | `float` from `0` to `1`; `0.95` | Drop one of a pair of features whose absolute correlation is greater than this threshold. |
| `max_missing_rate` | `float` from `0` to `1`; `0.30` | Drop columns with a missing-value rate above this limit before imputing remaining values. |
| `row_threshold_minibatch` | positive `int`; `50_000` | Use MiniBatchKMeans above this row count; use standard K-Means at or below it. Lower it to switch to minibatch clustering sooner. |
| `dr_reducer` | `bool`; `False` | If `True`, return `(normal_output, fitted_reducer)`. The reducer may be LDA, PCA, the supplied transformer, or `None` when supervised reduction is not applicable. |

Normal labeled split output is `(X_train, X_test, y_train, y_test)`; without labels it is
`(X_train, X_test)`. In outlier/inlier mode the outputs are `(X_inliers, X_outliers)` or
`(X_inliers, X_outliers, y_inliers, y_outliers)`. Each order is preserved when `dr_reducer=True`
by wrapping the normal output and reducer together.

### Other clustering functions

`m_attp_cluster` accepts a numeric pandas `DataFrame` or 2D NumPy array. It uses standardized
features and Gaussian-mixture clusters, returning `(X_train, X_test)`. Its optional `y` is accepted
for call compatibility but ignored; additional keyword arguments are also ignored.

| Function | Parameters beyond `X` | Meaning |
| --- | --- | --- |
| `m_attp_cluster` | `y: Series, DataFrame, ndarray, or None=None`, `test_size: float=0.2`, `random_state: Optional[int]=42`, `n_clusters: int=5`, `dr_reducer: bool=False`, `**kwargs` | Uses GMM cluster labels to stratify a split. `dr_reducer=True` returns `(normal_output, None)` because this method does not fit a dimensionality reducer. `y` and `**kwargs` are ignored. |
| `s_attp_cluster` | Same parameters as `s_attp_clustered_sampling` | Backward-compatible alias. Prefer `s_attp_clustered_sampling` for new code. |

### Clustering examples

```python
from attp import s_attp_clustered_sampling

# Labeled split: preprocessing, LDA, and class-by-cluster stratification.
X_train, X_test, y_train, y_test = s_attp_clustered_sampling(
    X, y, test_size=0.2, n_clusters=5
)

# Unlabeled split: preprocessing, PCA, and cluster-only stratification.
X_train, X_test = s_attp_clustered_sampling(X, test_size=0.2)

# Isolate the upper-distance tail as outliers.
X_inliers, X_outliers = s_attp_clustered_sampling(X, o_i=True)
```

The [ATTP Framework Reference Architecture Documentation](https://github.com/Jabualani/ATTP/blob/main/ATTP%20Framework%20Reference%20Architecture%20Documentation.pdf)
describes the library's concepts and variants. It is included in the GitHub repository and in the
installed `attp` package.

Set `dr_reducer=True` to append the fitted reducer to the normal result; it is `None` only when
supervised reduction cannot be applied (for example, a single target class).

The package also exports `auto_attp_clf`, `auto_attp_reg`, `auto_attp_ts`,
`auto_attp_al_clf`, `auto_attp_al_reg`, `mc_mcls_attp_reg`, `r_tf_attp`,
`r_ra_attp`, `r_d_attp`, `m_attp_ts`, and `m_attp_cluster`.

### Original function signature

`S_ATTP` keeps the original call signature for existing code:

```python
from attp import S_ATTP

X_tr, y_tr, X_te, y_te = S_ATTP(
    X_df, y_df, "target_col", [0.7, 0.8], 2, 10, 1, 3, None, "frames", False, False
)
```

`return_type="indices"` returns `(train_index, test_index)`, and
`return_split_metadata=True` appends the metrics DataFrame. If no valid split exists it returns
`None`s, like the original.

## Caveats

- LDA is fitted on the **whole** dataset (using all labels) before splitting, and the winning
  split is chosen by its **test** score. Both make the reported test score optimistic. Treat it as
  a way to build a representative split, not as an unbiased performance estimate.
- For `s_attp_clf`, features must be numeric with no missing or infinite values. The
  `s_attp_clustered_sampling` method instead preprocesses mixed types and imputes remaining
  missing values by default.

## License

MIT