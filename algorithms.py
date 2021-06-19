import numpy as np
import time
import logging
import os


def san(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1, dist=None):
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
    # Denote n = n_samples, d = n_features, we have model coefficients x \in R^d,
    # and we introduce n auxiliary variables {alpha_i} \in R^d, thus we have totally (n+1) variables.
    # We use a big matrix alphas \in \R^{nxd} to store auxiliary variables.
    n, d = data.shape
    alphas = np.zeros((n, d))  # auxiliary variables, it represents one alpha per row
    x = x_0.copy()  # model
    # initial gradient
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records, time_records = [np.sqrt(g @ g)], [0.0]

    cnt = 0  # track the effective data passes
    epoch_running_time, total_running_time = 0.0, 0.0
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution

    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == n:
            alphas = alphas - np.mean(alphas, axis=0, keepdims=True)  # update all alphas
        else:  # i \in [|0, n-1|]
            dot_i = data[i, :] @ x
            # second-order derivation of (i-1)-th loss
            dprime = loss.dprime(label[i], dot_i)
            diff = alphas[i, :] - loss.prime(label[i], dot_i) * \
                data[i, :] - reg * regularizer.prime(x)
            inv = 1. / (1. + reg * regularizer.dprime(x))
            scaled_data = inv * data[i, :]
            cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
            update = lr * (inv * diff - cte * scaled_data)
            alphas[i, :] -= update  # update i-th alpha
            x += update  # update x
        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + \
                reg * regularizer.prime(x)
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt,
                                                                                                  epoch_running_time,
                                                                                                  norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records
    return x, norm_records, time_records


def sana(loss, regularizer, data, label, reg, epoch, x_0, tol=None, verbose=1):
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients x \in R^d, we have totally (n+1) variables.
    # We use a big matrix alphas \in \R^{nxd} to store auxiliary variables.
    n, d = data.shape
    alphas = np.zeros((n, d))  # initialization, it represents one alpha per row
    x = x_0.copy()
    # g = np.mean((-label / (1 + np.exp(label * (data @ W[:, 0])))).reshape(-1, 1) * data, axis=0) + reg * W[:, 0]
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)

    norm_records = [np.sqrt(g @ g)]
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    iis = np.random.randint(0, n, n * epoch)  # uniform sampling
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()

        dot_i = data[i, :] @ x
        prime, dprime = loss.prime(label[i], dot_i), loss.dprime(label[i], dot_i)
        reg_prime, reg_dprime = regularizer.prime(x), regularizer.dprime(x)
        diff = alphas[i, :] - prime * data[i, :] - reg * reg_prime
        inv = 1. / (((n - 1) / n) + reg * reg_dprime)
        scaled_data = inv * data[i, :]
        cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
        update_w = inv * diff - cte * scaled_data
        new_alpha_i = (dprime * (data[i, :] @ update_w) + prime) * data[i, :] + \
            reg * (reg_dprime * update_w + reg_prime)
        alphas -= (1 / (n - 1)) * (new_alpha_i - alphas[i, :]).reshape(1, -1)
        alphas[i, :] = new_alpha_i
        x += update_w  # update w

        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + \
                reg * regularizer.prime(x)
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt,
                                                                                                  epoch_running_time,
                                                                                                  norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records
    return x, norm_records, time_records


def sag(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic average gradient algorithm.
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
                    "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt,
                                                                                                  epoch_running_time,
                                                                                                  norm_records[-1]))
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records

    return x, norm_records, time_records


def svrg(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic variance reduction gradient algorithm.

    reference: Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, Johnson & Zhang

    Note: for all stochastic methods, we measure the performance as a function of the number of effective passes
    through the data, measured as the number of queries to access single gradient (or Hessian) divided by
    the size of dataset. To have a fair comparison with others methods, for SVRG, we pay a careful attention
    to the step where we do a full pass of dataset at the reference point,
    it means that the effective passes should be added one after this step.
    """
    max_effective_pass = epoch // 2
    n, d = data.shape
    x = x_0.copy()
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records, time_records, total_running_time = [np.sqrt(g @ g)], [0.0], 0.0

    effective_pass = 0
    for idx in range(max_effective_pass):

        start_time = time.time()
        x_ref = x.copy()
        tot_grad = np.mean(loss.prime(label, data @ x_ref).reshape(-1, 1) * data, axis=0) + \
            reg * regularizer.prime(x_ref)
        x -= lr * tot_grad
        epoch_running_time = time.time() - start_time
        effective_pass += 1
        g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        norm_records.append(np.sqrt(g @ g))
        total_running_time += epoch_running_time
        time_records.append(total_running_time)
        if verbose == 1:
            logging.info(
                "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(effective_pass,
                                                                                              epoch_running_time,
                                                                                              norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, time_records

        iis = np.random.randint(low=0, high=n, size=n)
        epoch_running_time = 0.0

        for idx in range(len(iis)):  # inner loop
            i = iis[idx]
            start_time = time.time()
            grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
            grad_i_ref = loss.prime(label[i], data[i, :] @ x_ref) * data[i, :] + reg * regularizer.prime(x_ref)
            d_i = grad_i - grad_i_ref + tot_grad
            x -= lr * d_i
            epoch_running_time += time.time() - start_time

        effective_pass += 1
        g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        norm_records.append(np.sqrt(g @ g))
        total_running_time += epoch_running_time
        time_records.append(total_running_time)
        if verbose == 1:
            logging.info(
                "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(effective_pass,
                                                                                              epoch_running_time,
                                                                                              norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, time_records

    return x, norm_records, time_records


def snm(loss, data, label, reg, epoch, x_0, tol=None, verbose=1, path=None):
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

    if path and os.path.isfile(os.path.join(path, "Hessian_inv.npy")):
        B = np.load(os.path.join(path, "Hessian_inv.npy"))
        print("===SNM: load hessian inverse from local ======")
    else:
        B = np.linalg.pinv(reg * np.eye(d) + (H.T @ H) / n)
        np.save(os.path.join(path, "Hessian_inv.npy"), B)

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
                    "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt,
                                                                                                  epoch_running_time,
                                                                                                  norm_records[-1]))
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records

    return x, norm_records, time_records


def san_id(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1, dist=None):
    """
    Stochastic Average Newton method for linear model, projection under standard norm, e.g., W = I.
    Attention: bad performance in practice !!!
    """
    n, d = data.shape
    x = x_0.copy()
    alphas = np.zeros((n, d))  # auxiliary variables, it represents one alpha per row
    # W = np.zeros((n + 1, d))  # initialization
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * x
    data_norm = np.sum(data ** 2, axis=1)
    norm_records, time_records = [np.sqrt(g @ g)], [0.0]

    cnt = 0  # track the effective data passes
    epoch_running_time, total_running_time = 0.0, 0.0
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution

    for iter_cnt, i in enumerate(iis):

        start_time = time.time()
        if i == n:
            alphas = alphas - np.mean(alphas, axis=0, keepdims=True)  # update all alphas
        else:  # i \in [| 0, n-1 |]
            dot_i = data[i, :] @ x
            grad_i = loss.prime(label[i], dot_i) * data[i, :] + reg * x
            dprime = loss.dprime(label[i], dot_i)
            norm_i_square = data_norm[i]  # ||a_i||^2
            cte_i = (dprime * dprime * norm_i_square + 2 * reg * dprime) / (1 + reg ** 2)
            diff = alphas[i, :] - grad_i
            vec = (1 / (1 + reg ** 2)) * diff - \
                  (diff @ data[i] / (1 + reg ** 2)) * (cte_i / (1 + cte_i * norm_i_square)) * data[i]
            # update
            alphas[i, :] -= lr * vec  # update i-th alpha
            coef = dprime * (diff @ data[i]) / (1 + reg ** 2) - (dprime * norm_i_square * (diff @ data[i]) * cte_i) / ((1 + cte_i * norm_i_square) * (1 + reg ** 2))
            update_x = reg * vec + coef * data[i]
            x += lr * update_x  # update x

        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * x
            norm_records.append(np.sqrt(g @ g))
            total_running_time += epoch_running_time
            time_records.append(total_running_time)
            if verbose:
                logging.info(
                    "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt,
                                                                                                  epoch_running_time,
                                                                                                  norm_records[-1]))
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, time_records
    return x, norm_records, time_records


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
    for cnt in range(1, epoch + 1):

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
                "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
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
    for cnt in range(1, epoch + 1):

        start_time = time.time()

        grad = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        h = np.sqrt(loss.dprime(label, data @ x)).reshape(-1, 1) * data
        hess = reg * np.diag(regularizer.dprime(x)) + (h.T @ h) / n
        # x -= lr * np.linalg.lstsq(hess, grad, rcond=None)[0]
        x -= lr * np.linalg.solve(hess, grad)

        epoch_running_time = time.time() - start_time

        # evaluate
        g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        norm_records.append(np.sqrt(g @ g))
        total_running_time += epoch_running_time
        time_records.append(total_running_time)
        if verbose == 1:
            logging.info(
                "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(cnt, epoch_running_time,
                                                                                              norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, time_records

    return x, norm_records, time_records
