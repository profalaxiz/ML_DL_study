import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler

train_x = np.array([[1003, 2133, 321], 
                   [4321, 3214, 1999],
                   [2343, 1823, 1230]])
train_y = np.array([42, 21, 19])
train_y2 = np.array([0, 1, 1])

# z-score normalization
# linear regression model
scaler = StandardScaler()
X_norm = scaler.fit_transform(train_x)
# fit + train model
sgd = SGDRegressor(max_iter=1000)       # or i could use LinearRegression() for a closed-form solution
sgd.fit(X_norm, train_y)
print(f"number of iterations: {sgd.n_iter_}")
print(f"number of weight updates: {sgd.t_}")
w = sgd.coef_
b = sgd.intercept_
print(f"parameters after SGD: w = {w}, b = {b}")
y_pred = sgd.predict(X_norm)
print(f"target values: {train_y}")
print(f"predictions: {np.array2string(y_pred, precision=2)}")
print(f"R² score: {sgd.score(X_norm, train_y):.3f}")

# logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_norm, train_y2)
w2 = lr_model.coef_
b2 = lr_model.intercept_
print(f"lr w = {w2}, b = {b2}")
y2_pred = lr_model.predict(X_norm)
print(f"accuracy: {lr_model.score(train_x, train_y2)*100:.2f}%")


# # short summary
# x_normalized = scaler.fit_transform(x_train)
# model = SGDClassifier(max_iter=1000)
# model.fit(x_normalized, y_train)
# y_pred = model.predict(x_test_normalized)
# y_pred = (y_pred >= 0.5).astype(int)
# accuracy = model.score(x_train, y_train) * 100