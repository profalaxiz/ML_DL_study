import numpy as np

train_x = np.array([0, 1, 2, 3, 4, 5])
train_y = np.array([0, 0, 0, 1, 1, 1])

"""
sigmoid function: f(z) = 1 / 1+exp(-z)
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""
f_wb(x) = g(w.x + b)
"""
def hyp_func(x, w, b):
    # z = np.dot(x, w) + b
    m, n = x.shape
    f_wb = np.zeros(m)
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = x[i, j] * w[j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb[i] = sigmoid(z_wb) 
    return f_wb

w_test1 = 0.5
b_test1 = -1
f_wb_test1 = hyp_func(train_x, w_test1, b_test1)
print(f_wb_test1.tolist())

""""
loss(f_wb[i], y[i]) = -y[i]*log(f_wb[i]) - (1 - y[i])*log(1 - f_wb[i]); f_wb=sigmoid(wx+b)
cost: J(w,b) = 1/m * sigma(i=1, i=m)(loss(f_wb[i], y[i]))
"""
def cost_func(x, y, w, b):
    m = x.shape[0]
    f_wb = hyp_func(x, w, b)
    f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
    cost = 0.
    for i in range(m):
        loss = -y[i]*np.log(f_wb[i]) - (1 - y[i]) * np.log(1 - f_wb[i])
        cost += loss
    cost /= m
    return cost

train_x2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
train_y2 = np.array([0, 0, 0, 1, 1, 1])
w_test2 = np.array([1, 1])
b_test2 = -3
J_test1 = cost_func(train_x2, train_y2, w_test2, b_test2)
print(f"cost function (w = {w_test2.tolist()}, b = {b_test2}): J = {J_test1}")

def par_deriv(x, y, w, b):
    m = x.shape[0]  
    n = x.shape[1]
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
    for i in range(iterations):
        dj_dw, dj_db = par_deriv(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

w_final = np.zeros(train_x2.shape[1])
b_final = 0
iterations = 10000
alpha = 3.5
w_final, b_final = grad_desc(train_x2, train_y2, w_final, b_final, iterations, alpha)
print(f"after gradient descent: w = {w_final.tolist()}, b = {b_final:.2f}")
J_final = cost_func(train_x2, train_y2, w_final, b_final)
print(f"final cost function: J = {J_final:.3f}")

pred_probs = hyp_func(train_x2, w_final, b_final)
print(pred_probs)
print((pred_probs >= 0.5).astype(int))
print(train_y2)

## regularization

# L2 regularized cost function
def regul_cost_func(x, y, w, b, _lambda=1):
    m = x.shape[0]
    n = x.shape[1]
    f_wb = hyp_func(x, w, b)
    f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
    cost = 0.
    for i in range(m):
        loss = -y[i]*np.log(f_wb[i]) - (1 - y[i]) * np.log(1 - f_wb[i])
        cost += loss
    cost /= m
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = reg_cost * (_lambda / (2 * m))
    total_cost = cost + reg_cost
    return total_cost

J_test1_regul = regul_cost_func(train_x2, train_y2, w_test2, b_test2)
print(f"non regularized cost function: J = {J_test1:.5f}")
print(f"regularized cost function: J = {J_test1_regul:.5f}")

# L2 regularized gradient descent
def regul_par_deriv(x, y, w, b, lambda_):
    m = x.shape[0]  
    n = x.shape[1]
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
    for j in range(n):     
        dj_dw[j] += (lambda_ / m) * w[j]
    return dj_dw, dj_db 

def regul_grad_desc(x, y, w, b, iterations, alpha, lambda_):
    for i in range(iterations):
        dj_dw, dj_db = regul_par_deriv(x, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b