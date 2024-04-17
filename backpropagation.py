import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output
    
    def backward_propagation(self, inputs, targets, outputs):
        error = targets - outputs
        delta_output = error
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * self.learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += np.dot(inputs.T, delta_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            outputs = self.forward_propagation(inputs)
            self.backward_propagation(inputs, targets, outputs)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {np.mean(np.square(targets - outputs))}")

# Example usage:
# Define training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Initialize neural network
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 1000

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# Train the neural network
nn.train(X_train, y_train, epochs)

