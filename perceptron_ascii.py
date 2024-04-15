# prompt: Write a Python Program using Perceptron Neural Network to recognize even and odd numbers.
# Given numbers are in ASCII form 0 to 9

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def train(X, y, learning_rate=0.05, epochs=10000):
  w = np.random.randn(3)
  b = np.random.randn()

  for epoch in range(epochs):
    for i, x in enumerate(X):
      z = np.dot(x, w) + b
      y_pred = sigmoid(z)
      error = y[i] - y_pred
      w += learning_rate * error * x
      b += learning_rate * error

  return w, b

def predict(X, w, b):
  y_pred = []
  for x in X:
    z = np.dot(x, w) + b
    y_pred.append(sigmoid(z))

  return np.array(y_pred)

# Training data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([0, 1, 1, 0])

# Train the model
w, b = train(X, y)

# Test data
X_test = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

# Predict
y_pred = predict(X_test, w, b)

# Print the results
print(y_pred)

