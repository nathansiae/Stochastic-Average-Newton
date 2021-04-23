import numpy as np


class LogisticLoss:
    @staticmethod
    def val(y, y_hat):
        return np.log(1 + np.exp(-y * y_hat))

    @staticmethod
    def prime(y, y_hat):
        return -y / (1 + np.exp(y * y_hat))

    @staticmethod
    def dprime(y, y_hat):
        a = np.exp(y * y_hat)
        return a / ((1 + a) ** 2)


class L2:
    @staticmethod
    def val(y, y_hat):
        return (y - y_hat)**2 / 2.

    @staticmethod
    def prime(y, y_hat):
        return y_hat - y

    @staticmethod
    def dprime(y, y_hat):
        return np.ones_like(y_hat)


class PseudoHuberLoss:

    def __init__(self, delta=1.0):
        self.delta = delta

    def val(self, y, y_hat):
        diff = y_hat - y
        return (self.delta ** 2) * (np.sqrt(1. + (diff / self.delta) ** 2) - 1.)

    def prime(self, y, y_hat):
        diff = y_hat - y
        return diff / np.sqrt(1. + (diff / self.delta) ** 2)

    def dprime(self, y, y_hat):
        diff = y_hat - y
        return np.power((1. + (diff / self.delta) ** 2), -1.5)
