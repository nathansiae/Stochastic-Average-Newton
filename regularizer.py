import numpy as np


class L2:
    @staticmethod
    def val(x):
        return np.linalg.norm(x, ord=2)**2 / 2.

    @staticmethod
    def prime(x):
        return x

    @staticmethod
    def dprime(x):
        return np.ones_like(x)


class PseudoHuber:
    def __init__(self, delta=1.0):
        self.delta = delta

    def val(self, x):
        return np.sum((self.delta ** 2) * (np.sqrt(1. + (x / self.delta) ** 2) - 1.))

    def prime(self, x):
        return x / np.sqrt(1. + (x / self.delta) ** 2)

    def dprime(self, x):
        return np.power((1. + (x / self.delta) ** 2), -1.5)
