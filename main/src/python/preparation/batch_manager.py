from sklearn.model_selection import train_test_split
import numpy as np


class BatchManager:
    def __init__(self, X, y, batch_size=100):
        self.X_batches = X.reshape((-1, batch_size))
        self.y_batches = y.reshape((-1, batch_size))
        self.i = len(self.X_batches)
        self.reset_i = int(self.i)

    def has_next(self):
        return self.i > 0

    def num_batches(self):
        return len(self.X_batches)

    def next(self):
        self.i -= 1
        return self.X_batches[self.i], self.y_batches[self.i]

    def reset(self):
        self.i = self.reset_i
