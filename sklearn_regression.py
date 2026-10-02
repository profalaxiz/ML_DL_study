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

"""
model evaluation and selection
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X4 = np.linspace(0, 10, 100).reshape(-1, 1)
y4 = 1 + 2 * X4[:, 0] - 0.5 * X4[:, 0]**2 + np.random.randn(100) * 2
"""
split data into train/cross validation/test sets
training set: used to fit model parameters w, b
cross validation (dev set): used to choose model/hyperparameters (polynomial degree,lambda)
test set: only used at the end to estimate final model performance
"""
X_train, X_temp, y_train, y_temp = train_test_split(X4, y4, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"\n\ntotal samples: {len(X4)}")
print(f"training samples: {len(X_train)}")
print(f"CV samples: {len(X_cv)}")
print(f"test samples: {len(X_test)}")
"""
calculating error of a regression model
for model evaluation, calculate prediction error without including the regularization term
MSE = mean((yhat - y)**2)  
"""
def regression_error(model, X, y):
    yhat = model.predict(X)
    return mean_squared_error(y, yhat)
model4 = LinearRegression()
model4.fit(X_train, y_train)
train_error = regression_error(model4, X_train, y_train)
cv_error = regression_error(model4, X_cv, y_cv)
print(f"training error: {train_error}")
print(f"CV error: {cv_error}")
"""
choosing optimal polynomial degree
degree too small: model may underfit -> high bias
degree too large: model may overfit -> high variance
choose the degree with the lowest CV error
"""
max_degree = 9
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
for degree in range(1, max_degree + 1):
    model = make_pipeline(PolynomialFeatures(max_degree, include_bias=False), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    err_train[degree - 1] = regression_error(model, X_train, y_train)
    err_cv[degree - 1] = regression_error(model, X_cv, y_cv)
optimal_degree = np.argmin(err_cv) + 1
print(f"optimal polynomial degree: {optimal_degree}")
print(f"lowest CV error: {err_cv[optimal_degree - 1]:.3f}")
degrees = np.arange(1, max_degree + 1)
# plt.plot(degrees, err_train, marker="o", label="training error")
# plt.plot(degrees, err_cv, marker="o", label="CV error")
# plt.xlabel("polynomial degree")
# plt.ylabel("MSE (error)")
# plt.legend()
# plt.show()
"""
tuning regularization lambda
ridge regression uses L2 regularization, lambda in sklearn is "alpha"
small alpha: weak regularization, may lead to overfitting / high variance
large alpha: strong regularization, weights are pushed toward 0, may lead to underfitting / high bias
choose alpha using CV error
"""
lambda_range = np.array([
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    10,
    100
])
num_steps = len(lambda_range)
err_train_reg = np.zeros(num_steps)
err_cv_reg = np.zeros(num_steps)
for i in range(num_steps):
    lambda_ = lambda_range[i]
    model = make_pipeline(PolynomialFeatures(degree=optimal_degree, include_bias=False), StandardScaler(), Ridge(alpha=lambda_))
    model.fit(X_train, y_train)
    err_train_reg[i] = regression_error(model, X_train, y_train)
    err_cv_reg[i] = regression_error(model, X_cv, y_cv)
optimal_reg_idx = np.argmin(err_cv_reg)
optimal_lambda = lambda_range[optimal_reg_idx]
print(f"optimal lambda: {optimal_lambda}")
print(f"lowest CV error: {err_cv_reg[optimal_reg_idx]:.3f}")
"""
training final model and evaluating test error
after degree and lambda have been selected, the test set can finally be used
!do not use test error to choose degree or lambda!
"""
final_model = make_pipeline(PolynomialFeatures(degree=optimal_degree, include_bias=False), StandardScaler(), Ridge(alpha=optimal_lambda))
final_model.fit(X_train, y_train)
test_error = regression_error(final_model, X_test, y_test)
print(f"final test error: {test_error:.3f}")
"""
learning curve: increasing training set size
another diagnostic is to train the same model using, increasing amounts of training data
if adding more training data keeps reducing CV error, more data may help
usually:
    training error increases as training set gets larger
    CV error decreases as training set gets larger
"""
train_sizes = np.arange(10, len(X_train) + 100, 5)
err_train_size = np.zeros(len(train_sizes))
err_cv_size = np.zeros(len(train_sizes))
# shuffle first so that progressively larger subsets are random
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_train))
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]
for i, size in enumerate(train_sizes):
    X_subset = X_train_shuffled[:size]
    y_subset = y_train_shuffled[:size]
    model = make_pipeline(PolynomialFeatures(degree=optimal_degree, include_bias=False), StandardScaler(), Ridge(alpha=optimal_lambda))
    model.fit(X_subset, y_subset)
    err_train_size[i] = regression_error(model, X_subset, y_subset)
    err_cv_size[i] = regression_error(model, X_cv, y_cv)
plt.plot(train_sizes, err_train_size, marker="o", label="training error")
plt.plot(train_sizes, err_cv_size, marker="o", label="CV error")
plt.xlabel("training set size")
plt.ylabel("MSE (error)")
plt.legend()
plt.show()