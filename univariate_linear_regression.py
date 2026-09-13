import numpy as np
import matplotlib.pyplot as plt

train_x = np.array([1, 2, 3, 4])
train_y = np.array([300., 500., 700., 900.])

"""
f_wb[i] = x[i] * w + b
"""
def hyp_func(x, w, b):
    # return x * w + b
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = x[i] * w + b
    return f_wb

w_test1 = 200
b_test1 = 100
y_hat_test1 = hyp_func(train_x, w_test1, b_test1) 

# print(f"target values: {train_y}")
# print(f"predicted values: {y_hat}")

# plt.plot(train_x, y_hat_test1, c='r', label='prediction')
# plt.scatter(train_x, train_y, c='b', label='target')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Hypothesis function first test")
# plt.show()

"""
cost function (MSE): J(w,b) = 1/2m * sigma(i=1, i=m)(error)**2; error = y_hat[i] - y[i]
"""

def cost_func(x, y, w, b):
    m = x.shape[0]
    f_wb = hyp_func(x, w, b)
    j_wb = 0
    for i in range(m):
        error = f_wb[i] - y[i]
        j_wb += error ** 2
    j_wb /= (2 * m)
    return j_wb

# w_test2 = 100
# b_test2 = 50
# J_wb_test2 = cost_func(train_x, train_y, w_test2, b_test2)
# print(f"cost function (w = {w_test2}, b = {b_test2}): J = {J_wb_test2:.2f}")
# w_test3 = 199
# b_test3 = 99
# J_wb_test3 = cost_func(train_x, train_y, w_test3, b_test3)
# print(f"cost function (w = {w_test3}, b = {b_test3}): J = {J_wb_test3:.2f}")
# w_test4 = 200
# b_test4 = 100
# J_wb_test4 = cost_func(train_x, train_y, w_test4, b_test4)
# print(f"cost function (w = {w_test4}, b = {b_test4}): J = {J_wb_test4:.2f}")

"""
batch gradient descent

dj_dw = 1/m * sigma(i=1, i=m)(error)*x[i]
dj_db = 1/m * sigma(i=1, i=m)(error)

w = w - alpha*dj_dw
b = b - alpha*dj_db
"""
def par_deriv(x, y, w, b):
    m = x.shape[0]
    f_wb = hyp_func(x, w, b)
    dj_dw, dj_db = 0., 0.
    for i in range(m):
        error = f_wb[i] - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def grad_desc(x, y, w, b, iterations, alpha):
    J_history = [cost_func(x, y, w, b)]
    w_history = [w]
    for i in range(iterations):
        dj_dw, dj_db = par_deriv(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db   
        J_history.append(cost_func(x, y, w, b))
        w_history.append(w)
    return w, b, J_history, w_history

iterations_test1 = 1000
alpha_test1 = 1.0e-2  # learning rate
w_test5, b_test5 = 0., 0.
w_final, b_final, J_history, w_history = grad_desc(train_x, train_y, w_test5, b_test5, iterations_test1, alpha_test1)
print(f"after gradient descent: w = {w_final},   b = {b_final}")
plt.plot(J_history)
plt.xlabel("cost")
plt.ylabel("iterations")
plt.title("cost vs iteration graph")
plt.show()  

# print(f"after gradient descent: w = {w_final},   b = {b_final}")
# plt.scatter(train_x, train_y, c='b', label='target')
# f_wb_final = hyp_func(train_x, w_final, b_final)
# plt.plot(train_x, f_wb_final, c='r', label='prediction')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Plot after finding paramaters w and b with gradient descent")
# plt.show()
