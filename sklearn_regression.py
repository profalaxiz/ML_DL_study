import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

# simple linear regression model
X1 = np.array([[1], [2], [3], [4]])
y1 = np.array([300, 500, 700, 900])
model1 = LinearRegression()
model1.fit(X1, y1)
print(f"w = {model1.coef_}, b = {model1.intercept_}")
print(f"prediction for x = 5: {model1.predict([[5]])}")

# simple logistic regression model
X2 = np.array([[1], [2], [3], [4], [5]])
y2 = np.array([0, 0, 0, 1, 1])
model2 = LogisticRegression()
model2.fit(X2, y2)
print(f"w = {model2.coef_}, b = {model2.intercept_}")
print(f"class prediction for x = 2.5: {model2.predict([[2.5]])}")
print(f"probabilities prediction for x = 2.5: {model2.predict_proba([[2.5]])}")  # [P(class 0), P(class 1)]

# simple softmax regression model
X3 = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 1.0],
    [5.0, 8.0],
    [6.0, 9.0],
    [7.0, 8.0],
    [8.0, 2.0],
    [9.0, 3.0],
    [8.0, 4.0]
])
y3 = np.array([0, 0, 0,
              1, 1, 1,
              2, 2, 2])
K = 3    # 0, 1, 2
model3 = LogisticRegression()
model3.fit(X3, y3)
print(f"w = {model3.coef_}, b = {model3.intercept_}")
print(f"class prediction for [8.0, 2.0]: {model3.predict([[8.0, 2.0]])}")
print(f"probabilities prediction for x = [8.0, 2.0]: {model3.predict_proba([[8.0, 2.0]])}")

train_x = np.array([[1003, 2133, 321], 
                   [4321, 3214, 1999],
                   [2343, 1823, 1230]])
train_y = np.array([42, 21, 19])
train_y2 = np.array([0, 1, 1])

# z-score normalization
# linear regression model
"""
linear/logistic regression trained with gradient-based optimization can benefit greatly from scaling
standardization: z = (x - mean) / std deviation , features become centered around mean ~ 0 and std dev ~ 1
"""
scaler = StandardScaler()  
X_norm = scaler.fit_transform(train_x)
"""
fit_transform():
    fit: learn the mean and std deviation from training data
    transform: use them to scale training data
transform():
    do not fit again, used with test data
"""
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
