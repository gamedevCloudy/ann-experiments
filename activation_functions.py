# prompt: Write a Python program to plot a few activation functions that are being used in
# neural networks. Sigmoid, Tanh, Relu, leaky relu, softmax

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-8, 8, 1000)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x):
    return np.maximum(0.01 * x, x)

# Softmax
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")
plt.plot(x, softmax(x), label="Softmax")

plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

