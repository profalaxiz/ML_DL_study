import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True) #keepdims=True to keep the result as col so numpy can broadcast it
    return probs    

def hyp_func(X, W, b):
  # z = (m, n) @ (n, N) = (m, N) = (examples, classes)
    z = X @ W + b
    return softmax(z)      # for each row, softmax will convert each of those k scores into probabilies between 0 and 1 that sum up to 1 

""""
indicator function 1{y_i == j}: 1 if j is the correct class, 0 otherwise
J(w,b) = -1/m * sum(i=1, i=m)(sum(j=1, j=N)(indicator function * log(probs[i, j])))
"""
def cost_func(X, y, W, b):
    probs = hyp_func(X, W, b)
    cost = 0.0
    m = X.shape[0]
    N = W.shape[1]
    for i in range(m):  
        for j in range(N):  
            indicator = 1 if y[i] == j else 0
            cost += (-indicator * np.log(probs[i, j]))
    cost /= m
    return cost  # scalar

def par_deriv(X, y, W, b):
    probs = hyp_func(X, W, b)
    m, n = X.shape
    N = W.shape[1]
    dj_dW = np.zeros((n, N))
    dj_db = np.zeros(N)
    for i in range(m):  # current example
        for j in range(N):  # current class
            indicator = 1 if y[i] == j else 0
            error = probs[i, j] - indicator
            dj_db[j] += error
            for k in range(n):  # current feature
                dj_dW[k, j] += error * X[i, k]
    dj_dW /= m
    dj_db /= m
    return dj_dW, dj_db

def grad_desc(X, y, W, b, iterations, alpha):
    J_history = [cost_func(X, y, W, b)]
    for i in range(iterations):
        dj_dW, dj_db = par_deriv(X, y, W, b)
        W = W - alpha * dj_dW
        b = b - alpha * dj_db
        cost = cost_func(X, y, W, b)
        J_history.append(cost)
        if i % 100 == 0:
            print(
                f"Iteration {i:4d}: "
                f"Cost = {cost:.6f}"
            )
    return W, b, J_history

def predict(X, W, b):
    probs = hyp_func(X, W, b)  # (m, N)
    predictions = np.argmax(probs, axis=1) 
    return predictions # (m,)

X = np.array([
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
y = np.array([0, 0, 0,
              1, 1, 1,
              2, 2, 2])
num_classes = 3    # 0, 1, 2

m, n = X.shape
W_init = np.zeros((n, num_classes))
b_init = np.zeros(num_classes)

W_final, b_final, J_history = grad_desc(X, y, W_init, b_init, 4000, 0.05)
print(f"final W = ")
print(W_final)
print(f"final b: ")
print(b_final)

predictions = predict(X, W_final, b_final)
print(f"predictions: ")
print(predictions)   
print("actual: ")
print(y)
accuracy = np.mean(predictions == y) * 100
print(f"accuracy = {accuracy:.2f}%")

# plt.figure()
# plt.plot(J_history)
# plt.xlabel("Iteration")
# plt.ylabel("Cost J(W, b)")
# plt.title("Softmax Regression Training Cost")
# plt.show()

plt.figure()
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=predictions
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Softmax Regression Predictions")
plt.show()