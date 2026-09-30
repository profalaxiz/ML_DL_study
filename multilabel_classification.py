import numpy as np

"""
multilabel classification means one sample can belong to multiple classes at the same time
we use one sigmoid output per label, and train all labels with binary cross-entropy
why use sigmoid instead of softmax? softmax forces classes to compete and sum up to 1, sigmoid has no requirement for that
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""
shapes:
X shape = (m, n)
Y shape = (m, k)
W shape = (n, k)
b shape = (1, k) -> every output label gets its own bias
A_ij = P(Y_ij = 1 | X_i)
"""
def hyp_func(X, W, b):
    Z = X @ W + b
    return sigmoid(Z)
"""
we calculate binary cross-entropy for every label, so
"""
def cost_func(X, Y, W, b):
    m = X.shape[0]
    k = Y.shape[1]
    A = hyp_func(X, W, b)
    A = np.clip(A, 1e-15, 1 - 1e-15)
    cost = 0.0
    for i in range(m):          # each example
        for j in range(k):      # each label
            y = Y[i, j]
            a = A[i, j]
            cost += -(y * np.log(a) +
                      (1 - y) * np.log(1 - a))
    cost /= m
    return cost

def vectorized_cost_func(X, Y, W, b):
    f_wb = hyp_func(X, W, b)
    m = X.shape[0]
    f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
    cost = -np.sum(Y*np.log(f_wb) + (1 - Y)*np.log(1 - f_wb)) / m
    return cost

def par_deriv(X, Y, W, b):
    m, n = X.shape
    k = Y.shape[1]
    A = hyp_func(X, W, b)
    dj_dW = np.zeros((n, k))
    dj_db = np.zeros((1, k))
    for i in range(m):              # each example
        for j in range(k):          # each label
            error = A[i, j] - Y[i, j]
            dj_db[0, j] += error
            for f in range(n):      # each feature
                dj_dW[f, j] += error * X[i, f]
    dj_dW /= m
    dj_db /= m

def vectorized_par_der(X, Y, W, b):
    m = X.shape[0]
    f_wb = sigmoid(X @ W + b)
    f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
    errors = f_wb - Y
    dj_dw = (X.T @ errors) / m
    dj_db = np.sum(errors, axis=0, keepdims=True) / m
    return dj_dw, dj_db

def grad_desc(X, Y, W, b, iterations, alpha):
    cost_history = []    
    for i in range(iterations):
        cost = vectorized_cost_func(X, Y, W, b)
        dj_dw, dj_db = vectorized_par_der(X, Y, W, b)
        W -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history.append(cost)

        if i % 500 == 0:
            print(f"Epoch {i:4d} | Cost = {cost:.6f}")    
    return W, b, cost_history

def predict(X, W, b, threshold):
    probs = hyp_func(X, W, b)
    predictions = (probs >= threshold).astype(int)
    return predictions

# test
X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.8, 0.9],
    [0.9, 0.8],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.9, 0.1],
    [0.8, 0.2]
])

# 3 independent labels
# column 0: feature 1 is high
# column 1: feature 2 is high
# column 2: either feature is high
Y = np.array([
    [0, 0, 0],
    [0, 0, 0],

    [1, 1, 1],
    [1, 1, 1],

    [0, 1, 1],
    [0, 1, 1],

    [1, 0, 1],
    [1, 0, 1]
])

m, n = X.shape
k = Y.shape[1]
W = np.zeros((n, k))
b = np.zeros((1, k))
W, b, cost_history = grad_desc(X, Y, W, b, 5000, 0.5)
probs = hyp_func(X, W, b)
predictions = predict(X, W, b, 0.5)

print("Probabilities:")
print(np.round(probs, 3))
print("Predictions:")
print(predictions)
print("True labels:")
print(Y)
accuracy = np.mean(Y == predictions)
print(f"Label accuracy = {accuracy*100}%")