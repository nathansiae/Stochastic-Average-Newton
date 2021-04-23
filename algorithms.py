import numpy as np
import time
import logging


def san(loss, regularizer, data, label, lr, reg, dist, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic Average Newton method for linear model, projection under a norm that depends on hessian of f_i
    :param loss: loss function object, methods: val, prime, dprime
    :param regularizer: regularizer object, methods: val, prime, dprime
    :param data: numpy array, shape (n_samples, n_features)
    :param label: numpy array, shape (n_samples,)
    :param lr: float, learning rate
    :param reg: float, non-negative
    :param dist: sketching distribution
    :param epoch: int, number of data pass
    :param x_0: numpy array of shape (d,), initial point
    :param tol: float, the algo will be stopped if the norm of gradient is less than this threshold
    :param verbose: 0 or 1; 0 means silence, no events be logged;
    :return: trained model params, a list of gradients' norm
    """
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients, we have totally (n+1) variables. We use a matrix W \in \R^{(n+1)xd} to store these variables.
    # W[0, :] stores the model, and W[i:0] stores the i-th alpha, where i \in [|1, n|]
    n, d = data.shape
    W = np.zeros((d, n + 1))  # initialization
    W[:, 0] = x_0.copy()
    # g = np.mean((-label / (1 + np.exp(label * (data @ W[:, 0])))).reshape(-1, 1) * data, axis=0) + reg * W[:, 0]
    g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(W[:, 0])

    norm_records = [np.sqrt(g @ g)]
    # loss = [np.mean(np.log(1 + np.exp(-label * (data @ W[:, 0]))))]  # records empirical loss during training
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    # iis = np.random.randint(0, n + 1, n * epoch)  # uniform sampling
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == 0:
            W[:, 1:] = W[:, 1:] - lr * np.mean(W[:, 1:], axis=1, keepdims=True)  # update all alphas
        else:  # i \in [|1, n|]
            dot_i = data[i - 1, :] @ W[:, 0]
            # second-order derivation of (i-1)-th loss
            dprime = loss.dprime(label[i - 1], dot_i)
            diff = W[:, i] - loss.prime(label[i - 1], dot_i) * data[i - 1, :] - reg * regularizer.prime(W[:, 0])
            inv = 1. / (1. + reg * regularizer.dprime(W[:, 0]))
            scaled_data = inv * data[i - 1, :]
            cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i - 1, :] @ scaled_data))
            update = lr * (inv * diff - cte * scaled_data)
            W[:, i] -= update  # update i-th alpha
            W[:, 0] += update  # update x
        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + \
                reg * regularizer.prime(W[:, 0])
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return W[:, 0], norm_records, time_records
    return W[:, 0], norm_records, time_records


def sana(loss, regularizer, data, label, reg, epoch, x_0, tol=None, verbose=1):
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients, we have totally (n+1) variables. We use a matrix W \in \R^{(n+1)xd} to store these variables.
    # W[0, :] stores the model, and W[i:0] stores the i-th alpha, where i \in [|1, n|]
    n, d = data.shape
    W = np.zeros((d, n + 1))  # initialization
    W[:, 0] = x_0.copy()
    # g = np.mean((-label / (1 + np.exp(label * (data @ W[:, 0])))).reshape(-1, 1) * data, axis=0) + reg * W[:, 0]
    g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(W[:, 0])

    norm_records = [np.sqrt(g @ g)]
    # loss = [np.mean(np.log(1 + np.exp(-label * (data @ W[:, 0]))))]  # records empirical loss during training
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    iis = np.random.randint(0, n, n * epoch)  # uniform sampling
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()

        dot_i = data[i, :] @ W[:, 0]
        prime, dprime = loss.prime(label[i], dot_i), loss.dprime(label[i], dot_i)
        reg_prime, reg_dprime = regularizer.prime(W[:, 0]), regularizer.dprime(W[:, 0])
        diff = W[:, i + 1] - prime * data[i, :] - reg * reg_prime
        inv = 1. / (((n - 1) / n) + reg * reg_dprime)
        scaled_data = inv * data[i, :]
        cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
        update_w = inv * diff - cte * scaled_data
        new_alpha_i = (dprime * (data[i, :] @ update_w) + prime) * data[i, :] + reg * (reg_dprime * update_w + reg_prime)
        W[:, 1:] = W[:, 1:] - (1 / (n - 1)) * (new_alpha_i - W[:, i + 1]).reshape(-1, 1)
        W[:, i + 1] = new_alpha_i
        W[:, 0] += update_w  # update w

        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + \
                reg * regularizer.prime(W[:, 0])
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return W[:, 0], norm_records, time_records
    return W[:, 0], norm_records, time_records


def svrg2(loss, regularizer, data, label, lr, reg, dist, epoch, x_0, tol=None, verbose=1):
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients, we have totally (n+1) variables. We use a matrix W \in \R^{(n+1)xd} to store these variables.
    # W[0, :] stores the model, and W[i:0] stores the i-th alpha, where i \in [|1, n|]
    n, d = data.shape
    W = np.zeros((d, n + 1))  # initialization
    W[:, 0] = x_0.copy()
    W[:, 1:] = loss.prime(label, data @ W[:, 0]).reshape(1, -1) * data.T + \
               reg * regularizer.prime(W[:, 0]).reshape(-1, 1)
    avg_alpha = np.mean(W[:, 1:], axis=1)
    g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(W[:, 0])
    norm_records = [np.sqrt(g @ g)]
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    # iis = np.random.randint(0, n + 1, n * epoch)  # uniform sampling
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == 0:
            W[:, 1:] = loss.prime(label, data @ W[:, 0]).reshape(1, -1) * data.T + \
                       reg * regularizer.prime(W[:, 0]).reshape(-1, 1)
            avg_alpha = np.mean(W[:, 1:], axis=1)
        else:  # i \in [|1, n|]
            dot_i = data[i - 1, :] @ W[:, 0]
            # second-order derivation of (i-1)-th loss
            dprime = loss.dprime(label[i - 1], dot_i)
            diff = loss.prime(label[i - 1], dot_i) * data[i - 1, :] + reg * regularizer.prime(W[:, 0]) - \
                   W[:, i] + avg_alpha
            inv = 1. / reg * regularizer.dprime(W[:, 0])
            scaled_data = inv * data[i - 1, :]
            cte = lr * dprime * (scaled_data @ diff) / (1 + dprime * (data[i - 1, :] @ scaled_data))
            update = lr * inv * diff - cte * scaled_data
            W[:, 0] -= update  # update x
        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ W[:, 0]).reshape(-1, 1) * data, axis=0) + \
                reg * regularizer.prime(W[:, 0])
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return W[:, 0], norm_records, time_records
    return W[:, 0], norm_records, time_records


def sag(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic average gradient algorithm.
    This function is adapted from the code provided by Rui
    """
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    # Old gradients
    gradient_memory = np.zeros((n, d))
    y = np.zeros(d)

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # gradient of (i-1)-th loss
        grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        # update
        # x -= lr * (grad_i - gradient_memory[i] + y)  not exactly same with Algorithm 1 in SAG paper
        y += grad_i - gradient_memory[i]
        x -= lr * y / n
        gradient_memory[i] = grad_i
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose == 1:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records

    return x, norm_records, time_records


def svrg(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic variance reduction gradient algorithm.
    This function is adapted from the code provided by Rui
    """
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    x_old = x.copy()

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):

        # Update
        i = iis[idx]

        start_time = time.time()
        if idx % n == 0:
            x_old = x.copy()
            tot_grad = np.mean(loss.prime(label, data @ x_old).reshape(-1, 1) * data, axis=0) + \
                       reg * regularizer.prime(x_old)
            grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
            grad_i_old = loss.prime(label[i], data[i, :] @ x_old) * data[i, :] + reg * regularizer.prime(x_old)
            g_i = grad_i - grad_i_old + tot_grad
            x -= lr * g_i
        elif idx % n != 0:
            grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
            grad_i_old = loss.prime(label[i], data[i, :] @ x_old) * data[i, :] + reg * regularizer.prime(x_old)
            g_i = grad_i - grad_i_old + tot_grad
            x -= lr * g_i
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose == 1:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records

    return x, norm_records, time_records


def snm(loss, data, label, reg, epoch, x_0, tol=None, verbose=1):
    """
    This part implements the method introduced in the paper
    'Stochastic Newton and Cubic Newton Methods with Simple Local Linear-Quadratic Rates, Kovalev et al.',
    according to the Algorithm 3 (\tau = 1) for GLM case.
    Notice that this method supports only L2 regularizer and generally it has O(d^2) complexity.
    The trick used to lead an efficient Algorithm 3 can not be extended to
    other types of regularizer whose hessian are not identity.
    """
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * x
    norm_records = [np.sqrt(g @ g)]

    memory_gamma = data @ x
    memory_alpha, memory_beta = loss.prime(label, memory_gamma), loss.dprime(label, memory_gamma)

    g_ = np.mean(memory_alpha.reshape(-1, 1) * data, axis=0)
    h = np.mean((memory_beta * memory_gamma).reshape(-1, 1) * data, axis=0)
    H = np.sqrt(memory_beta).reshape(-1, 1) * data
    B = np.linalg.pinv(reg * np.eye(d) + (H.T @ H) / n)

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):

        # Update
        i = iis[idx]

        start_time = time.time()

        x = B @ (h - g_)
        gamma = data[i, :] @ x
        alpha, beta = loss.prime(label[i], gamma), loss.dprime(label[i], gamma)
        # update g_, h, B
        g_ += (alpha - memory_alpha[i]) * data[i, :] / n
        h += (beta * gamma - memory_beta[i] * memory_gamma[i]) * data[i, :] / n
        Ba = B @ data[i, :]
        B -= (beta - memory_beta[i]) / (n + (beta - memory_beta[i]) * (data[i, :] @ Ba)) * (
                Ba.reshape(-1, 1) @ Ba.reshape(1, -1))
        # update memory
        memory_gamma[i], memory_alpha[i], memory_beta[i] = gamma, alpha, beta

        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * x
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose == 1:
                logging.info(
                    "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                         norm_records[-1]))
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records

    return x, norm_records, time_records


def vsn(func, data, label, reg, epoch, x_0, tol=None, verbose=1):
    # TODO
    return


#########################
# Deterministic Algorithm
#########################


def gd(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Vanilla Gradient Descent
    """
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]

    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for cnt in range(1, epoch+1):

        start_time = time.time()

        grad = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        x -= lr * grad

        epoch_running_time = time.time() - start_time

        # evaluate
        g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        norm_records.append(np.sqrt(g @ g))
        total_running_time += epoch_running_time
        time_records.append(total_running_time)
        if verbose == 1:
            logging.info(
                "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                     norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, time_records

    return x, norm_records, time_records


def newton(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Vanilla Newton's Method
    """
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]

    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for cnt in range(1, epoch+1):

        start_time = time.time()

        grad = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        h = np.sqrt(loss.dprime(label, data @ x)).reshape(-1, 1) * data
        hess = reg * np.diag(regularizer.dprime(x)) + (h.T @ h) / n
        # x -= lr * np.linalg.lstsq(hess, grad, rcond=None)[0]
        x -= lr * np.linalg.pinv(hess) @ grad

        epoch_running_time = time.time() - start_time

        # evaluate
        g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        norm_records.append(np.sqrt(g @ g))
        total_running_time += epoch_running_time
        time_records.append(total_running_time)
        if verbose == 1:
            logging.info(
                "| end of epoch {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                     norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, time_records

    return x, norm_records, time_records
