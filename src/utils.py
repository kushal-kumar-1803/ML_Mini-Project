import numpy as np
import pandas as pd
import os


# Create required directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


def ensure_dir(path):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


# Simple scaler for normalization
class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-8

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_
