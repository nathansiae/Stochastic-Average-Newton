from abc import ABCMeta, abstractmethod
import numpy as np
import time
import logging
import os
import regularizer as my_reg


class Solver(metaclass=ABCMeta):

    def __init__(self, loss, regularizer):
        self.loss = loss
        self.regularizer = regularizer
        self.x = None
        self.norm_records = []
        self.fval_records = []
        self.time_records = [0.0]
        self.total_running_time = 0.0

    def run(self, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
        self.x = x_0.copy()
        # init grad and loss value
        g = np.mean(self.loss.prime(label, data @ self.x).reshape(-1, 1) * data, axis=0) \
            + reg * self.regularizer.prime(self.x)
        fval = np.mean(self.loss.val(label, data @ self.x)) + reg * self.regularizer.val(self.x)
        self.norm_records.append(np.sqrt(g @ g))
        self.fval_records.append(fval)
        self.auxiliary_proc(data)

        for cnt in range(1, epoch + 1):
            start_time = time.time()
            self.run_epoch(data, label, reg, lr)
            epoch_running_time = time.time() - start_time
            self.total_running_time += epoch_running_time
            # make records
            self.callback(data, label, reg)
            if verbose == 1:
                logging.info("| end of effective pass {:d} | time: {:f}s | \
                             norm of gradient {:f} |".format(cnt, epoch_running_time, self.norm_records[-1]))
            if tol is not None and self.norm_records[-1] <= tol:
                return self.x, self.norm_records, self.time_records, self.fval_records
        return self.x, self.norm_records, self.time_records, self.fval_records

    def auxiliary_proc(self, data):
        return

    @abstractmethod
    def run_epoch(self, data, label, reg, lr):
        pass

    def callback(self, data, label, reg):
        # current gradient and loss value
        g = np.mean(self.loss.prime(label, data @ self.x).reshape(-1, 1) * data, axis=0) \
            + reg * self.regularizer.prime(self.x)
        self.norm_records.append(np.sqrt(g @ g))
        fval = np.mean(self.loss.val(label, data @ self.x)) + reg * self.regularizer.val(self.x)
        self.fval_records.append(fval)
        self.time_records.append(self.total_running_time)
        return


class SAN(Solver):

    def __init__(self, loss, regularizer, customized_dist):
        super(SAN, self).__init__(loss, regularizer)
        self.dist = customized_dist  # customized sampling distribution
        self.alphas = None  # auxiliary variables, it represents one alpha per row

    def auxiliary_proc(self, data):
        n, d = data.shape
        self.alphas = np.zeros((n, d))
        return

    def run_epoch(self, data, label, reg, lr):
        n, d = data.shape
        iis = np.random.choice(n + 1, size=n, p=self.dist)  # sampling w.r.t customized distribution
        for i in iis:
            if i == n:
                self.alphas -= np.mean(self.alphas, axis=0, keepdims=True)  # update all alphas
            else:  # i \in [|0, n-1|]
                dot_i = data[i, :] @ self.x
                # second-order derivation of (i-1)-th loss
                dprime = self.loss.dprime(label[i], dot_i)
                diff = self.alphas[i, :] - self.loss.prime(label[i], dot_i) * \
                       data[i, :] - reg * self.regularizer.prime(self.x)
                inv = 1. / (1. + reg * self.regularizer.dprime(self.x))
                scaled_data = inv * data[i, :]
                cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
                update = lr * (inv * diff - cte * scaled_data)
                self.alphas[i, :] -= update  # update i-th alpha
                self.x += update  # update x
        return


class SAG(Solver):

    def __init__(self, loss, regularizer):
        super(SAG, self).__init__(loss, regularizer)
        # Old gradients
        self.gradient_memory = None
        self.y = None

    def auxiliary_proc(self, data):
        n, d = data.shape
        self.gradient_memory = np.zeros((n, d))
        self.y = np.zeros(d)
        return

    def run_epoch(self, data, label, reg, lr):
        n, d = data.shape
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            # gradient of (i-1)-th loss
            grad_i = self.loss.prime(label[i], data[i, :] @ self.x) * data[i, :] \
                     + reg * self.regularizer.prime(self.x)
            # update
            # x -= lr * (grad_i - gradient_memory[i] + y)  not exactly same with Algorithm 1 in SAG paper
            self.y += grad_i - self.gradient_memory[i]
            self.x -= lr * self.y / n
            self.gradient_memory[i] = grad_i
        return


class SVRG(Solver):
    """
    Stochastic variance reduction gradient algorithm.

    reference: Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, Johnson & Zhang

    Note: for all stochastic methods, we measure the performance as a function of the number of effective passes
    through the data, measured as the number of queries to access single gradient (or Hessian) divided by
    the size of dataset. To have a fair comparison with others methods, for SVRG, we pay a careful attention
    to the step where we do a full pass of dataset at the reference point,
    it means that the effective passes should be added one after this step.
    """

    def __init__(self, loss, regularizer):
        super(SVRG, self).__init__(loss, regularizer)
        self.x_ref = None
        self.grad_ref = None
        self.do_inner_loop = False

    def run_epoch(self, data, label, reg, lr):
        if self.do_inner_loop:
            self._inner_loop(data, label, reg, lr)
            self.do_inner_loop = False
        else:
            self._full_pass(data, label, reg, lr)
            self.do_inner_loop = True
        return

    def _full_pass(self, data, label, reg, lr):
        self.x_ref = self.x.copy()
        self.grad_ref = np.mean(self.loss.prime(label, data @ self.x).reshape(-1, 1) * data, axis=0) + \
                        reg * self.regularizer.prime(self.x)
        self.x -= lr * self.grad_ref
        return

    def _inner_loop(self, data, label, reg, lr):
        n, d = data.shape
        iis = np.random.randint(low=0, high=n, size=n)
        for i in iis:
            grad_i = self.loss.prime(label[i], data[i, :] @ self.x) * data[i, :] \
                     + reg * self.regularizer.prime(self.x)
            grad_i_ref = self.loss.prime(label[i], data[i, :] @ self.x_ref) * data[i, :] \
                         + reg * self.regularizer.prime(self.x_ref)
            d_i = grad_i - grad_i_ref + self.grad_ref
            self.x -= lr * d_i
        return


class GradientDescent(Solver):
    def run_epoch(self, data, label, reg, lr):
        grad = np.mean(self.loss.prime(label, data @ self.x).reshape(-1, 1) * data, axis=0) \
               + reg * self.regularizer.prime(self.x)
        self.x -= lr * grad
        return


class Newton(Solver):
    def run_epoch(self, data, label, reg, lr):
        n, d = data.shape
        grad = np.mean(self.loss.prime(label, data @ self.x).reshape(-1, 1) * data, axis=0) \
               + reg * self.regularizer.prime(self.x)
        h = np.sqrt(self.loss.dprime(label, data @ self.x)).reshape(-1, 1) * data
        hess = reg * np.diag(self.regularizer.dprime(self.x)) + (h.T @ h) / n
        # x -= lr * np.linalg.lstsq(hess, grad, rcond=None)[0]
        self.x -= lr * np.linalg.solve(hess, grad)
        return


# ==================================
# Extra methods proposed in the paper: https://arxiv.org/abs/2106.10520
# ==================================
class SANA(Solver):

    def __init__(self, loss, regularizer):
        super(SANA, self).__init__(loss, regularizer)
        self.alphas = None  # auxiliary variables, it represents one alpha per row

    def auxiliary_proc(self, data):
        n, d = data.shape
        self.alphas = np.zeros((n, d))
        return

    def run_epoch(self, data, label, reg, lr):
        n, d = data.shape
        iis = np.random.randint(low=0, high=n, size=n)  # uniform sampling
        for i in iis:
            dot_i = data[i, :] @ self.x
            prime, dprime = self.loss.prime(label[i], dot_i), self.loss.dprime(label[i], dot_i)
            reg_prime, reg_dprime = self.regularizer.prime(self.x), self.regularizer.dprime(self.x)
            diff = self.alphas[i, :] - prime * data[i, :] - reg * reg_prime
            inv = 1. / (((n - 1) / n) + reg * reg_dprime)
            scaled_data = inv * data[i, :]
            cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
            update_w = inv * diff - cte * scaled_data
            new_alpha_i = (dprime * (data[i, :] @ update_w) + prime) * data[i, :] + \
                reg * (reg_dprime * update_w + reg_prime)
            self.alphas -= (1 / (n - 1)) * (new_alpha_i - self.alphas[i, :]).reshape(1, -1)
            self.alphas[i, :] = new_alpha_i
            self.x += update_w  # update w
        return


class SANid(Solver):

    def __init__(self, loss, regularizer, customized_dist):
        super(SANid, self).__init__(loss, regularizer)
        self.dist = customized_dist  # customized sampling distribution
        self.regularizer = my_reg.L2()  # only L2 is supported for this method
        self.alphas = None  # auxiliary variables, it represents one alpha per row
        self.data_norm = 0.0

    def auxiliary_proc(self, data):
        n, d = data.shape
        self.alphas = np.zeros((n, d))
        self.data_norm = np.sum(data ** 2, axis=1)
        return

    def run_epoch(self, data, label, reg, lr):
        n, d = data.shape
        iis = np.random.choice(n + 1, size=n, p=self.dist)  # sampling w.r.t customized distribution
        for i in iis:
            if i == n:
                self.alphas -= np.mean(self.alphas, axis=0, keepdims=True)  # update all alphas
            else:  # i \in [| 0, n-1 |]
                dot_i = data[i, :] @ self.x
                grad_i = self.loss.prime(label[i], dot_i) * data[i, :] + reg * self.x
                dprime = self.loss.dprime(label[i], dot_i)
                norm_i_square = self.data_norm[i]  # ||a_i||^2
                cte_i = (dprime * dprime * norm_i_square + 2 * reg * dprime) / (1 + reg ** 2)
                diff = self.alphas[i, :] - grad_i
                vec = (1 / (1 + reg ** 2)) * diff - \
                      (diff @ data[i] / (1 + reg ** 2)) * (cte_i / (1 + cte_i * norm_i_square)) * data[i]
                # update
                self.alphas[i, :] -= lr * vec  # update i-th alpha
                coef = dprime * (diff @ data[i]) / (1 + reg ** 2) - (
                            dprime * norm_i_square * (diff @ data[i]) * cte_i) / (
                               (1 + cte_i * norm_i_square) * (1 + reg ** 2))
                update_x = reg * vec + coef * data[i]
                self.x += lr * update_x  # update x
        return
