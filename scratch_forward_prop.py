import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(a_in, w) + b[j]
        a_out[j] = g(z)
    return a_out

def dense_v(A_in, W, B, g):
    return g(A_in @ W + B)

def sequential(x, W1, b1, W2, b2):
    a1 = dense(x, W1, b1, sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    return (a2)

def predict(X, W1, b1, W2, b2):
    m = X.shape[0] 
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = sequential(X[i], W1, b1, W2, b2)[0]
    return (p)


a = np.array([1, 2]).reshape(2,)
w = np.array([3, 4]).reshape(2,)
print(f"a   .  w  =   {np.dot(a, w)}")
print(f"a.T .  w  =   {np.dot(a.T, w)}")
