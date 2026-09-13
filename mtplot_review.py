import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(10)
# y = x ** 2

# line plot -> visualize a model's predictions and training/validation curves (learning curve/time series/accuracy over epochs/loss over epochs/regression line)
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("x²")
# plt.title("Example")
# plt.show()

# scatter plot ->visualize/explore the dataset (raw observations, outliers, actual vs predicted)
# X = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11])
# Y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78])
# plt.scatter(X, Y, c='r')     
# plt.title("Analysis")
# plt.xlabel("X label")
# plt.ylabel("Y label")
# plt.show()

#practice
x = np.linspace(-10, 10, 100)
#1
y = x
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y = x")
plt.show()
#2
y = x ** 2
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y = x²")
plt.show()
#3
y = 1/(1 + np.exp(-x))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y (sigmoid)")
plt.show()
#4
data = np.random.normal(size=1000)
plt.hist(data, bins='auto')
plt.title("Histogram of 1000 random normal numbers")
plt.xlabel("value")
plt.ylabel("frequency")
plt.show()  
#5
x = np.linspace(0, 10, 100)
y = 3*x + 5 + np.random.randn(100) * 3
plt.scatter(x, y)
plt.plot(x, 3*x + 5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()  
