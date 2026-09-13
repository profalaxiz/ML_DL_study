import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.activations import sigmoid, relu, softmax # type: ignore
from sklearn.datasets import make_blobs

# explicit way of carrying out a forward prop one layer of computation at a time

# x = np.array([[200.0, 17.0]])
# layer1 = tf.keras.layers.Dense(units=3, activation="sigmoid")
# a1 = layer1(x)
# layer2 = tf.keras.layers.Dense(units=1, activation="sigmoid")
# a2 = layer2(a1)

# a more cleaner way

# x = np.array([
#     [200.0, 17.0],
#     [120.0, 5.0],
#     [425.0, 20.0],
#     [212.0, 18.0]
# ])
# y = np.array([1, 0, 0, 1])
# model = Sequential([
#     Dense(units=3, activation="sigmoid"),
#     Dense(units=1, activation="sigmoid")])


# ## linear model - neuron without activation

# # training data
# X_train = np.array([[1.0], [2.0]], dtype=np.float32)
# Y_train = np.array([[300.0], [500.0]], dtype=np.float32)
# # one neuron with linear function (w * x + b)
# linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
# # call the layer once so tensorflow creates w, b
# linear_layer(X_train)
# # initialize w, b
# w, b = linear_layer.get_weights()
# print(f"initial w = {w}")
# print(f"initial b = {b}")
# # manually set parameters
# set_w = np.array([[200]], dtype=np.float32)
# set_b = np.array([100], dtype=np.float32)
# linear_layer.set_weights([set_w, set_b])
# print(f"new weights: {linear_layer.get_weights()}")
# # prediction for x = 1
# a1 = linear_layer(X_train[0].reshape(1, 1))
# print(f"prediction for x = 1: {a1.numpy()}")
# # prediction for all examples
# predictions = linear_layer(X_train)
# plt.scatter(X_train, Y_train, c='b', label='Target')
# plt.plot(X_train, predictions, c='r', label='Predictions')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# # plt.show()  

# ## neuron with sigmoid activation

# X_train2 = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  
# Y_train2 = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  
# pos = Y_train2 == 1
# neg = Y_train2 == 0
# print(X_train2[pos])
# model2 = Sequential([
#     Dense(units=1, activation="sigmoid", name="L1")
# ])
# model2(X_train2)
# model2.summary()
# logistic_layer = model2.get_layer('L1')
# # w2, b2 = logistic_layer.get_weights()
# # print(f"w = {w2}, b = {b2}")
# # print(f"w : {w2.shape}, b : {b2.shape}")
# set_w = np.array([[2]])
# set_b = np.array([-4.5])
# logistic_layer.set_weights([set_w, set_b])
# w2, b2 = logistic_layer.get_weights()
# print(f"w = {w2}, b = {b2}")
# print(f"w : {w2.shape}, b : {b2.shape}")
# predict1 = model2.predict(X_train2[0].reshape(1, 1))
# print(f"prediction for x = 1: {predict1}")


# X = np.random.randint(1, 10, (10, 4), )
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(15, activation="relu"),
#     tf.keras.layers.Dense(5, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
# # output = model(X)
# # print(f"final output = {output}")
# a = X
# for layer in model.layers:
#     a = layer(a)
#     print(f"\n{layer.name}")
#     print(f"output shape: {a.shape}")
#     print(f"weight shape: {layer.kernel.shape}")
#     print(f"bias shape: {layer.bias.shape}")

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train3, y_train3 = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
print(f"input shape: {X_train3.shape}")
softmx_model = Sequential([ tf.keras.Input(shape=(2,)),
    Dense(25, activation='relu', name="L1"),
    Dense(15, activation='relu', name="L2"),
    Dense(4, activation='linear', name="Output")
])
"""
cross entropy loss numerical stability:
L(z) = C + log(sum(i=1, i=N)(e^(z_i-C))) - z_target, C = max(z)
"""
softmx_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # sparse means to give the class label directly, instead of one-hot encoding by CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.001),
)
softmx_model.fit(X_train3, y_train3, epochs=10)
a = X_train3
for layer in softmx_model.layers:
    a = layer(a)
    print(f"{layer.name} - output shape: {a.shape}")
    print(f"weight shape: {layer.kernel.shape}")
    print(f"bias shape: {layer.bias.shape}")
pds = softmx_model.predict(X_train3)  # logits, not probabilities
softmx_model.summary()
print(f"two example output vectors:\n {pds[:2]}")
print("largest value", np.max(pds), "smallest value", np.min(pds))
sm_pds = tf.nn.softmax(pds).numpy()   # only when you want interpretable probabilities (class probabilities)
print(f"two example output vectors:\n {sm_pds[:2]}")
print("largest value", np.max(sm_pds), "smallest value", np.min(sm_pds))
for i in range(5):
    print( f"{pds[i]}, category: {np.argmax(pds[i])}")   # argmax(logit) → predicted class




# # short summary
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import Sequential
# from tensorflow.keras.activations import relu, sigmoid, linear
# from tensorflow.keras.losses import BinaryCrossentropy , MeanSquaredError
# from tensorflow.keras.optimizers.legacy import Adam, SGD

# model_ = Sequential([
#     Dense(units=20, activation='linear'),
#     Dense(units=5, activation='relu'),
#     Dense(units=1, activation='sigmoid')
# ])
# model_.compile(
#     loss = BinaryCrossentropy(),
#     optimizer = Adam(0.001),
# )
# model_.fit(
#     train_x, train_y,
#     epochs = 20
# )
# predictions_ = model_.predict(...)

