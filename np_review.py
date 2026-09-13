import numpy as np

# a = np.array([[[1, 2, 3]],      # 3D array(layers, rows, cols)
#               [[5, 6, 7]]])
# b = np.array([[1, 2],
#               [3, 4]])
# zeros = np.zeros((3, 4))    # zeros(shape)
# ones = np.ones((2, 3))
# random = np.random.randint(1, 10, (3, 3))   # randint(low, high, size)
# x = np.arange(0, 10, 2)     # 10 is exclusive
# y = np.linspace(0, 1, 10)

# print(a)
# print(b)
# print(zeros)
# print(ones)
# print(random)
# print(x)
# print(y)
# print(a.shape), print(b.shape)
# print(b.ndim), print(a.ndim)
# print(a.size), print(b.size)
# print(a.dtype)
# print(random.dtype)


# for ML, shape (100, 20) -> 100 samples, 20 features

# X_matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# print(X_matrix[:, 0])
# print(X_matrix[0:2, :])     # 2 is exclusive
# print(X_matrix[X_matrix > 4]) # boolean indexing
# print((X_matrix[0:2, :]) >= 5)   

# x_origin = np.array([1, 2, 3])  # (3, )
# print(x_origin.reshape(-1, 1))
# print(a.flatten())  # flattens to 1d array/vector
# print(b.T)  # rows -> cols, cols -> rows
# (a, 1) col vector
# (1, a) row vector

# c = np.array([[1, 2],
#              [3, 4]])
# d = np.array([[1, 2],
#              [3, 4]])
# # element-wise operations
# print(c + d)
# print(c - d)
# print(c * d)
# print(c / d)
# print(c ** 2)
# print(np.sqrt(c))
# print(np.exp(c))
# print(np.log(c))

# #matrix multiplication
# print(c @ d)
# print(np.matmul(c, d))

# vectorization -> perform operations on entire array at once instead of iterating
# results = c ** 2 + 1
# print(results)
# aggregations -> any data transformation that produces scalar values from arrays
# print(np.sum(results))
# print(np.mean(results))
# print(np.std(results))  
# print(np.min(results))
# print(np.max(results))
# print(np.argmax(results))  # flattens the array, returns the index of max value   
# print(np.argmin(results))
# print(results.mean(axis=0))  # gives the mean of each feature
# print(results.mean(axis=1))  # gives the mean of each sample

# broadcasting -> automatic expansion of array dimensions to be of equal sizes and shapes without copying array data to perform arithmetic operations
# the smaller array automatically stretches/replicates across the larger one

# #task 1:
# X = np.arange(1, 25)
# #1
# X_res = X.reshape(6, 4)
# print(X_res)
# #2
# print(X_res[:, 2].reshape(-1,1))  
# #3
# print(X_res[4:, ])
# #4
# #X_res[X_res > 15] = 0   -> modifies the original array
# X_final = np.where(X_res > 15, 0, X_res)  # where(condition, value if true, value if false)
# print(X_final)
# #5
# mean_vals = X_res.mean(axis=0)
# print(mean_vals)

# #task 2:
# A = np.random.randn(10, 5) * 10 + 50
# print(f"original matrix:\n{A}")
# A = (A - A.mean(axis=0)) / A.std(axis=0)  # formula to normalize A
# print(f"after normalization:\n{A}")
# print(A.mean(axis=0))
# print(A.std(axis=0))    

# #task 3:
# E = np.random.randint(2, 10, (100, 3))
# w = np.array([3, 5, 9])
# b = 2
# y = E @ np.array([2, 4, 8]) + 5 + np.random.randn(100) # (100, ) + (100, )
# print(f"y shape = {y.shape}")
# y_hat = E @ w + b
# # print(y_hat)
# print(f"y_hat shape = {y_hat.shape}")
# mse = np.mean((y - y_hat)**2) 

# #task 4:
# O = np.array([-2, -1, 0, 1, 2])
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))
# print(sigmoid(O))

# list/dict practice
# words = ["machine", "learning", "AI", "python", "neural", "network"]
# #1
# length = [len(x) for x in words]
# print(length)
# #2
# words_updated = [x for x in words if len(x) >= 6]
# print(words_updated)
# #3
# upper_words = [x.upper() for x in words]
# print(upper_words)
# #4
# dict_words = {word:len(word) for word in words}
# print(dict_words)
# #5
# enum_words = {i:word for i,word in enumerate(words)}
# print(enum_words)
