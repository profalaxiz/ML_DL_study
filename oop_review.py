import numpy as np

#task 1
class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def num_samples(self):
        return self.x.shape[0]
    def num_features(self):
        return self.x.shape[1]

X = np.random.randn(100, 5)
Y = np.random.randn(100)
dataset = Dataset(X, Y)
print(dataset.num_samples())
print(dataset.num_features())

#task 2
class LinearModel:
    def __init__(self, weights, bias):
        self.w = weights
        self.b = bias
    def predict(self, x):
        return x @ self.w + self.b

w = np.random.randint(1, 4, (5))
b = 2
model = LinearModel(w, b)
predictions = model.predict(X)
print(predictions)

