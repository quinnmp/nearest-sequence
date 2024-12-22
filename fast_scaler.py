from numba import jit
import numpy as np
import torch

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
    def _transform_numpy(X, mean, scale):
        return (X - mean) / scale

    @staticmethod
    @jit(nopython=True)
    def _inverse_transform_numpy(X, mean, scale):
        return X * scale + mean

    def _transform_torch(self, X, mean, scale):
        mean_tensor = torch.tensor(mean, dtype=X.dtype, device=X.device)
        scale_tensor = torch.tensor(scale, dtype=X.dtype, device=X.device)
        return (X - mean_tensor) / scale_tensor

    def _inverse_transform_torch(self, X, mean, scale):
        mean_tensor = torch.tensor(mean, dtype=X.dtype, device=X.device)
        scale_tensor = torch.tensor(scale, dtype=X.dtype, device=X.device)
        return X * scale_tensor + mean_tensor

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            return self._transform_torch(X, self.mean_, self.scale_)
        else:
            if isinstance(X, list):
                X = np.array(X)
            return self._transform_numpy(X, self.mean_, self.scale_)

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            return self._inverse_transform_torch(X, self.mean_, self.scale_)
        else:
            if isinstance(X, list):
                X = np.array(X)
            return self._inverse_transform_numpy(X, self.mean_, self.scale_)
