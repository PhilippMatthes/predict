from sklearn.model_selection import train_test_split
import numpy as np


class BatchManager:
    def __init__(self, X, y, batch_size=100, test_size=0.2, random_state=42):
        self.X_batches = np.array_split(X, batch_size)
        self.y_batches = np.array_split(y, batch_size)
        self.i = len(self.X_batches)
        self.test_size = test_size
        self.random_state = random_state

    def has_next(self):
        return self.i > 0

    def num_batches(self):
        return len(self.X_batches)

    def next(self):
        self.i -= 1
        return train_test_split(self.X_batches[self.i], self.y_batches[self.i], test_size=self.test_size, random_state=self.random_state)

    def reset(self):
        self.i = len(self.X_batches)
