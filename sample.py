from sklearn.datasets import load_iris
from attp import sattp_split

data = load_iris(as_frame=True)
result = sattp_split(
    data.data,
    data.target,
    splits=[0.7],
    min_clusters=2,
    max_clusters=4,
)

x_train, y_train, x_test, y_test = result
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("test score:", result.test_score)