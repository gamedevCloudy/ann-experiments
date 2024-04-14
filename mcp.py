# prompt: generate AndNot function using mucclouch pits model using numpy, show outputs

import numpy as np

# Define the McCulloch-Pitts model
def mcp(x, w, b):
  return np.where(x.dot(w) + b > 0, 1, 0)

# Define the NOT function
def not_func(x):
  w = np.array([-1])
  b = np.array([0.5])
  return mcp(x, w, b)

# Generate input data
x = np.array([[0], [1]])

# Calculate the output of the NOT function
y = not_func(x)

# Print the inputs and outputs
print("Inputs:")
print(x)
print("Outputs:")
print(y)

