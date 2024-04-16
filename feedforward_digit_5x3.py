import numpy as np
import tensorflow as tf

# Define training data
training_data = {
    0: [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
    1: [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    2: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
    3: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    4: [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
    5: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    6: [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
    7: [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
    8: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
    9: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
}



# Define labels for training data
labels = np.eye(10)  # One-hot encoding for numbers 0 to 9

# Convert training data to numpy arrays
X_train = np.array([np.array(num).flatten() for num in training_data.values()])
y_train = np.array([labels[num] for num in training_data.keys()])

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(15,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000)

# Test the model
test_data = [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]]  # Example test data for number 2
X_test = np.array([np.array(test_data).flatten()])
predictions = model.predict(X_test)
recognized_number = np.argmax(predictions)

print("Recognized number:", recognized_number)

