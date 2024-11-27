from numba import jit
import numpy as np

class FastScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        if isinstance(X, list):
            X = np.array(X)
        self.mean_ = np.mean(X, axis=0, dtype=np.float64)
        self.scale_ = np.std(X, axis=0, dtype=np.float64)

        if np.ndim(self.scale_) == 0:
            self.scale_ = np.array([self.scale_])

        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1
        return self

    @staticmethod
    @jit(nopython=True)
    def _transform(X, mean, scale):
        return (X - mean) / scale

    @staticmethod
    @jit(nopython=True)
    def _inverse_transform(X, mean, scale):
        return X * scale + mean

    def transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        return self._transform(X, self.mean_, self.scale_)

    def inverse_transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        return self._inverse_transform(X, self.mean_, self.scale_)
