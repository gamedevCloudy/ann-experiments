import numpy as np

class ART:
    def __init__(self, num_input, num_output, vigilance_parameter):
        self.num_input = num_input
        self.num_output = num_output
        self.vigilance_parameter = vigilance_parameter
        
        # Initialize weights and reset states
        self.weights = np.random.rand(num_output, num_input)
        self.reset()
    
    def reset(self):
        self.activation_state = np.zeros(self.num_output)
    
    def normalize(self, x):
        return x / np.sum(x)
    
    def calculate_similarity(self, input_pattern, output_pattern):
        return np.dot(input_pattern, output_pattern) / np.sum(output_pattern)
    
    def predict(self, input_pattern):
        self.reset()
        while True:
            max_activation = np.max(self.activation_state)
            if max_activation == 0:
                break
            
            winner_index = np.argmax(self.activation_state)
            winner_pattern = self.weights[winner_index]
            similarity = self.calculate_similarity(input_pattern, winner_pattern)
            
            if similarity >= self.vigilance_parameter:
                return winner_index
            
            self.activation_state[winner_index] = 0
    
    def train(self, input_patterns):
        for input_pattern in input_patterns:
            for i in range(self.num_output):
                similarity = self.calculate_similarity(input_pattern, self.weights[i])
                if similarity >= self.vigilance_parameter:
                    self.activation_state[i] += 1
                    self.weights[i] = self.normalize(self.weights[i] + input_pattern)
                    break

# Define input patterns
input_patterns = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0]
])

# Initialize and train ART network
art = ART(num_input=4, num_output=3, vigilance_parameter=0.5)
art.train(input_patterns)

# Test the trained network
test_patterns = np.array([
    [1, 1, 0, 0],  # Close to pattern 1
    [0, 1, 0, 1],  # Close to pattern 3
    [0, 0, 1, 1],  # Close to pattern 2
    [1, 0, 1, 0]   # Close to pattern 1 and pattern 3
])

for pattern in test_patterns:
    winner_index = art.predict(pattern)
    print(f"Input pattern: {pattern}, Winner index: {winner_index}")

