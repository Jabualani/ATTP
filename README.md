# sattp

Cluster-stratified train/test splitting on an LDA-reduced feature space (S-ATTP).

Instead of stratifying on the class label alone, `sattp` also stratifies on *where a row sits in
feature space*, so the test set covers the same sub-populations as the train set.

## How it works

1. Reduce the features with **LDA**, using the fewest components that reach 85% of the explained
   variance (configurable).
2. For each candidate cluster count, run **K-Means** on the LDA components
   (`MiniBatchKMeans` above 50,000 rows).
3. Build strata from `class × cluster` and do a stratified split. Strata with a single member
   can't be split, so those rows always go to the train set.
4. Score a classifier on every `(train fraction, n_clusters)` pair and return the split with the
   best test score.

## Install

```bash
pip install git+https://github.com/<owner>/<repo>.git
```

For development:

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
pip install -e ".[dev]"
pytest
```

## Usage

```python
from sattp import sattp_split

result = sattp_split(
    X, y,
    splits=[0.7, 0.75, 0.8],   # candidate train fractions
    min_clusters=2,
    max_clusters=10,
)

X_train, y_train, X_test, y_test = result      # unpacks like a 4-tuple
print(result.best_split, result.best_n_clusters, result.n_lda_components)
print(result.metrics.head())                    # every candidate, best first
```

Row labels of the original `X`/`y` are preserved in the outputs.

Useful options: `model=` (any scikit-learn classifier used for scoring; defaults to LDA),
`scoring=` (custom `(fitted_model, X, y) -> float`), `variance_threshold=`, `random_state=`,
`verbose=True`.

### Original function signature

`S_ATTP` keeps the original call signature for existing code:

```python
from sattp import S_ATTP

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
- Features must be numeric with no missing values.

## License

MIT