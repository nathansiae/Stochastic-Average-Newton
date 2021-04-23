import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import pickle


def plot_grad_iter(result_dict, title, save_path, threshold=1e-8):
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.figure(figsize=(9, 8), dpi=1200)

    markers = ["^-", "d-", "*-", ">-", "+-", "o-"]
    for algo_name, marker in zip(sorted(result_dict.keys()), markers):
        result = result_dict[algo_name]
        # result is a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
        len_cut = len(min(result, key=len))
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        # plot
        grad_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = np.argmax(grad_avg <= threshold) + 1 if np.sum(grad_avg <= threshold) > 0 else len(grad_avg)
            grad_avg = grad_avg[:len_cut]
        grad_min = np.min(result, axis=0)[:len(grad_avg)]
        grad_max = np.max(result, axis=0)[:len(grad_avg)]
        # grad_std = np.std(result, axis=0)
        # grad_min = np.add(grad_avg, -grad_std)
        # grad_max = np.add(grad_avg, grad_std)
        plt.semilogy(np.arange(len(grad_avg)), grad_avg, marker, label=algo_name, lw=2)
        plt.fill_between(np.arange(len(grad_avg)), grad_min, grad_max, alpha=0.2)

    plt.tick_params(labelsize=20)
    plt.legend(fontsize=30)
    plt.xlabel("Effective Passes", fontsize=25)
    plt.ylabel(r"$\| \nabla f \|_2$", fontsize=25)
    plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + "_grad_iter.pdf"), bbox_inches='tight', pad_inches=0.01)


def plot_grad_time(result_dict, title, save_path):
    # TODO
    return


def run_algorithm(algo_name, algo, algo_kwargs, n_repeat):
    logging.info("------START {}------".format(algo_name))
    grad_iter, grad_time = [], []
    for i in range(n_repeat):
        logging.info("{}-th repetition:".format(i + 1))
        final_w, norm, times = algo(**algo_kwargs)
        grad_iter.append(norm)
        grad_time.append(times)
    logging.info("------END {}------".format(algo_name))
    # print("MAE: {}".format(np.mean(np.abs(algo_kwargs['data']@final_w - algo_kwargs['label']))))
    return grad_iter, grad_time


def save(path_grad_iter, grad_iter, path_grad_time, grad_time):
    with open(path_grad_iter, 'wb') as fp:
        pickle.dump(grad_iter, fp)
    with open(path_grad_time, 'wb') as fp:
        pickle.dump(grad_time, fp)


def load(path_grad_iter, path_grad_time):
    grad_iter, grad_time = None, None
    if os.path.isfile(path_grad_iter):
        with open(path_grad_iter, 'rb') as fp:
            grad_iter = pickle.load(fp)
    if os.path.isfile(path_grad_time):
        with open(path_grad_time, 'rb') as fp:
            grad_time = pickle.load(fp)
    return grad_iter, grad_time


def lipschitz_ridge(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / n + reg


def lipschitz_logistic(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / (4. * n) + reg


def max_Li_ridge(X, reg):
    return np.max(np.sum(X ** 2, axis=1)) + reg


def max_Li_logistic(X, reg):
    return 0.25*np.max(np.sum(X ** 2, axis=1)) + reg
