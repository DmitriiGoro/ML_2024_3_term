import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        return np.mean((X.dot(w) - Y) ** 2)

    @staticmethod
    def mae(X, Y, w):
        return np.mean(np.abs(X.dot(w) - Y))

    @staticmethod
    def l2_reg(w):
        return np.sum(np.square(w))

    @staticmethod
    def l1_reg(w):
        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):
        return 0.0

    @staticmethod
    def mse_derivative(X, Y, w):
        n_observations = X.shape[0]

        predictions = np.dot(X, w)

        error = predictions - Y

        if Y.ndim > 1:
          target_dimensionality = Y.shape[1]
          return (2 / (n_observations * target_dimensionality)) * np.dot(X.T, error)

        return (2 / n_observations) * np.dot(X.T, error)

    @staticmethod
    def mae_derivative(X, Y, w):
        if Y.ndim > 1:
            return (1 / (X.shape[0] * Y.shape[1])) * np.dot(X.T, np.sign(np.dot(X, w) - Y))

        return (1 / X.shape[0]) * np.dot(X.T, np.sign(np.dot(X, w) - Y))

    @staticmethod
    def l2_reg_derivative(w):
        return 2 * w


    @staticmethod
    def l1_reg_derivative(w):
        return np.sign(w)

    @staticmethod
    def no_reg_derivative(w):
        return np.zeros_like(w)