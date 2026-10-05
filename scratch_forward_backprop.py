import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def der_relu(z):
    return 1 if z > 0 else 0

def dense_forward(a_in, W, b, g):
    units = W.shape[1]
    z_out = np.zeros(units)
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(a_in, w) + b[j]
        z_out[j] = z
        a_out[j] = g(z)
    return z_out, a_out

def sequential_forward(x, W1, b1, W2, b2, W3, b3):
    z1, a1 = dense_forward(x, W1, b1, relu)
    z2, a2 = dense_forward(a1, W2, b2, relu)
    z3, a3 = dense_forward(a2, W3, b3, sigmoid)
    return z1, a1, z2, a2, z3, a3

def sequential_backprop(x, y, W1, W2, W3, cache):
    z1, a1, z2, a2, z3, a3 = cache
    dW3 = np.zeros_like(W3)
    db3 = np.zeros(W3.shape[1])
    dW2 = np.zeros_like(W2)
    db2 = np.zeros(W2.shape[1])
    dW1 = np.zeros_like(W1)
    db1 = np.zeros(W1.shape[1])
    dz3 = np.zeros_like(a3)
    for j in range(len(a3)):
        dz3[j] = a3[j] - y[j]
    for i in range(W3.shape[0]):
        for j in range(W3.shape[1]):
            dW3[i, j] = a2[i] * dz3[j]
    for j in range(len(db3)):
        db3[j] = dz3[j]
    da2 = np.zeros(len(a2))
    for i in range(len(a2)):
        for j in range(len(dz3)):
            da2[i] += W3[i, j] * dz3[j]
    dz2 = np.zeros(len(z2))
    for j in range(len(z2)):
        dz2[j] = da2[j] * der_relu(z2[j])
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            dW2[i, j] = a1[i] * dz2[j]
    for j in range(len(db2)):
        db2[j] = dz2[j]
    da1 = np.zeros(len(a1))
    for i in range(len(a1)):
        for j in range(len(dz2)):
            da1[i] += W2[i, j] * dz2[j]
    dz1 = np.zeros(len(z1))
    for j in range(len(z1)):
        dz1[j] = da1[j] * der_relu(z1[j])
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            dW1[i, j] = x[i] * dz1[j]
    for j in range(len(db1)):
        db1[j] = dz1[j]
    return dW1, db1, dW2, db2, dW3, db3

def binary_cross_entropy(y, a):
    a = np.clip(a, 1e-15, 1 - 1e-15)
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))

def train(X, Y, W1, b1, W2, b2, W3, b3, learning_rate=0.01, epochs=1000):
    m = X.shape[0]
    for epoch in range(epochs):
        dW1_total = np.zeros_like(W1)
        db1_total = np.zeros_like(b1)
        dW2_total = np.zeros_like(W2)
        db2_total = np.zeros_like(b2)
        dW3_total = np.zeros_like(W3)
        db3_total = np.zeros_like(b3)
        total_loss = 0
        for sample in range(m):
            x = X[sample]
            y = Y[sample]
            cache = sequential_forward(x, W1, b1, W2, b2, W3, b3)
            z1, a1, z2, a2, z3, a3 = cache
            total_loss += binary_cross_entropy(y, a3)[0]
            dW1, db1_grad, dW2, db2_grad, dW3, db3_grad = sequential_backprop(x, y, W1, W2, W3, cache)
            dW1_total += dW1
            db1_total += db1_grad
            dW2_total += dW2
            db2_total += db2_grad
            dW3_total += dW3
            db3_total += db3_grad
        dW1_total /= m
        db1_total /= m
        dW2_total /= m
        db2_total /= m
        dW3_total /= m
        db3_total /= m
        average_loss = total_loss / m
        W1 -= learning_rate * dW1_total
        b1 -= learning_rate * db1_total
        W2 -= learning_rate * dW2_total
        b2 -= learning_rate * db2_total
        W3 -= learning_rate * dW3_total
        b3 -= learning_rate * db3_total
        if epoch % 100 == 0:
            print(f"epoch {epoch}, loss = {average_loss:.6f}")
    return W1, b1, W2, b2, W3, b3

def predict(X, W1, b1, W2, b2, W3, b3):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        z1, a1, z2, a2, z3, a3 = sequential_forward(X[i], W1, b1, W2, b2, W3, b3)
        p[i, 0] = a3[0]
    return (p >= 0.5).astype(int)
