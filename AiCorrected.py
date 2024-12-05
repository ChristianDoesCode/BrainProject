import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from data.csv
data = pd.read_csv('data.csv')

# Separate features and labels
X = data.drop('label', axis=1).values
y = data['label'].values.astype(np.int8)

# Normalize pixel values
X = X / 255.0

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)

# Use a subset of the training data for faster computation
train_size = 10000
X_train = X_train[:train_size]
y_train = y_train[:train_size]

# Initialize weights
np.random.seed(42)
W = np.random.randn(784, 10) * 0.01

# Learning rate
eta = 0.01

# Number of epochs
num_epochs = 10

n_samples = X_train.shape[0]

for epoch in range(num_epochs):
    correct = 0
    for i in range(n_samples):
        x_i = X_train[i]  # Input vector
        y_i = y_train[i]  # True label

        # Compute the scores
        s_i = x_i.dot(W)  # Scores for each class
        y_pred_i = np.argmax(s_i)  # Predicted label

        if y_pred_i == y_i:
            # Correct prediction, strengthen the connections
            W[:, y_pred_i] += eta * x_i
            correct += 1
        else:
            # Incorrect prediction, weaken the connections
            W[:, y_pred_i] -= eta * x_i

    # Normalize weights to prevent them from growing too large
    W = W / np.linalg.norm(W, axis=0, keepdims=True)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {correct / n_samples * 100:.2f}%")

# Evaluate on test set
s_test = X_test.dot(W)
y_pred_test = np.argmax(s_test, axis=1)
accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
