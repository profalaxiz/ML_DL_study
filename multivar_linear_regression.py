import numpy as np
import matplotlib.pyplot as plt

train_x = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40], 
                    [852, 2, 1, 35]])
train_y = np.array([460., 232., 178.])

"""
f_wb[i] = dot(x[i], w) + b
"""
def hyp_func(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = np.dot(x[i], w) + b
    return f_wb
    # return x @ w + b    

w_test1 = np.array([0.39, 18.75, -53, -26.])
b_test1 = 785.18
f_wb_test1 = hyp_func(train_x, w_test1, b_test1)
print(f"target values: {train_y.tolist()}")
print(f"predicted values: {f_wb_test1.tolist()}")

"""
J(w,b) = 1/2m * sigma(i=1, i=m)(error)**2
"""
def cost_func(x, y, w, b):
    m = x.shape[0]
    f_wb = hyp_func(x, w, b)
    cost = 0.
    for i in range(m):
        error = f_wb[i] - y[i]
        cost += error**2
    cost /= (2 * m)
    return cost

J_wb_test1 = cost_func(train_x, train_y, w_test1, b_test1)
print(f"cost function (w = {w_test1.tolist()}, b = {b_test1}): J = {J_wb_test1:.2f}")

"""
dj_dw = 1/m * sigma(i=1, i=m)(sigma(i=1, i=n)error*x[i,j])
dj_db = 1/m * sigma(i=1, i=m)(error)
w = w - alpha * dj_dw
b = b - alpha * dj_db
"""
def par_deriv(x, y, w, b):
    m, n = x.shape[0], x.shape[1]
    f_wb = hyp_func(x, w, b)
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        error = f_wb[i] - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def grad_desc(x, y, w, b, iterations, alpha):
    J_history = [cost_func(x, y, w, b)]
    for i in range(iterations):
        dj_dw, dj_db = par_deriv(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(cost_func(x, y, w, b))
    return w, b, J_history

alpha_test1 = 1.0e-7
iterations = 4000
b_final = 0.
w_final = np.zeros(train_x.shape[1])
w_final, b_final, J_history = grad_desc(train_x, train_y, w_final, b_final, iterations, alpha_test1)

print(f"after gradient descent: w = {w_final.tolist()}, b = {b_final}")
# plt.plot(J_history)
# plt.xlabel("iterations") 
# plt.ylabel("cost")
# plt.title("cost vs iterations")
# plt.show() 

# J_final = cost_func(train_x, train_y, w_final, b_final)
# print(f"cost function (w = {w_final.tolist()}, b = {b_final}): J = {J_final:.2f}")

## feature scaling

# z-score normalization
mu = np.mean(train_x, axis=0)
sigma = np.std(train_x, axis=0)
train_x_scaled = (train_x - mu) / sigma    # broadcasting
print(train_x_scaled)
w_final2 = np.zeros(train_x_scaled.shape[1])
b_final2 = 0.
w_final2, b_final2, J_history2 = grad_desc(train_x_scaled, train_y, w_final2, b_final2, 1000, 0.5)
plt.plot(J_history2)
plt.xlabel("iterations") 
plt.ylabel("cost")
plt.title("cost vs iterations")
plt.show() 
print(f"after feature scaling and gradient descent: w = {w_final2.tolist()}, b = {b_final2}")
J_final = cost_func(train_x_scaled, train_y, w_final2, b_final2)
print(f"final cost function: {J_final}")

f_wb_final = hyp_func(train_x_scaled, w_final2, b_final2)
print(f"final predictions: {f_wb_final.tolist()}")