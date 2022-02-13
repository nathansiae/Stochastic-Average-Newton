import os
import argparse
import logging
import math
import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import load_data
import utils
import loss
import regularizer
import config
import solvers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', dest='data_set',
                        help="data set name")
    parser.add_argument('--data_path', action='store', dest='data_path',
                        help='path to load data')
    opt = parser.parse_args()
    data_set = opt.data_set
    folder_path = os.path.join(config.output_path, data_set)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.basicConfig(filename=os.path.join(folder_path, config.log_file),
                        level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    # logging.info(config)

    # ===========START: prepare data=============================================
    X, y = load_data.get_data(opt.data_path)
    X = X.toarray()  # convert from scipy sparse matrix to dense if necessary
    logging.info("Data Sparsity: {}".format(load_data.sparsity(X)))

    if config.problem_type == 'classification':
        # label preprocessing for some dataset whose label is not {-1, 1}
        max_v, min_v = np.max(y), np.min(y)
        idx_min = (y == min_v)
        idx_max = (y == max_v)
        y[idx_min] = -1
        y[idx_max] = 1
    elif config.problem_type == 'regression':
        X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
    else:
        raise Exception("Unknown problem type!")
    # adding a column filling with all ones.(bias term)
    X = np.c_[X, np.ones(X.shape[0])]

    n, d = X.shape
    logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))
    # =========END=======================================================================

    # =========START: set loss, regularizer, etc ========================================
    if config.loss == "L2":
        criterion = loss.L2()
    elif config.loss == "PseudoHuber":
        criterion = loss.PseudoHuberLoss(delta=1.0)
    elif config.loss == "Logistic":
        criterion = loss.LogisticLoss()
    else:
        raise Exception("Unknown loss function!")

    if config.regularizer == "L2":
        penalty = regularizer.L2()
    elif config.regularizer == "PseudoHuber":
        penalty = regularizer.PseudoHuber(delta=1.0)
    else:
        raise Exception("Unknown regularizer type!")

    if config.reg is None:
        if config.ill_conditional == 3:
            reg = 1. / (n * n)  # ill conditional
        elif config.ill_conditional == 1:
            reg = 1. / math.sqrt(n)
        else:
            reg = 1. / n
    else:
        reg = config.reg
    logging.info("Regularization param: {}".format(reg))

    epochs = config.epochs
    n_rounds = config.n_repetition
    x_0 = np.zeros(d)  # initial point for all algorithm, np.random.randn(d)
    # =============END====================================================================

    # =============START: sub-optimality==================================================
    # try to obtain the optimum by using l-bfgs method
    # but only the L2 logistic regression is supported
    f_opt = 0.0
    if config.subopt and config.loss == "Logistic" and config.regularizer == "L2":
        print("Running scipy fmin_l_bfgs_b --------")
        optimum, f_opt, d_info = fmin_l_bfgs_b(func=utils.f_val_logistic, x0=x_0, fprime=utils.f_grad_logistic,
                                               args=(X, y, criterion, penalty, reg), pgtol=1e-07)
        print("scipy fmin_l_bfgs_b finished------")

        # verify that the gradient at optimum is close to zero
        g_opt = d_info['grad']
        if np.sqrt((g_opt @ g_opt)) > 1e-5:
            print("The gradient at given optimum is larger than 1e-5, we think it is not an optimum")
            f_opt = 0.0
    # =============END====================================================================

    # for some algorithms, we may choose non-uniform sampling technique
    if config.is_uniform:
        dist = None
    else:
        # customized distribution
        p_0 = 1. / (n + 1)
        logging.info("Probability p_0: {:}".format(p_0))
        dist = np.array([p_0] + [(1 - p_0) / n] * n)

    # ==============START: run algorithms================================================
    algorithms2run, dict_algo_norm, dict_algo_time, dict_algo_fval = {}, {}, {}, {}
    if "SAN" in config.algorithms2run:
        algorithms2run["SAN"] = solvers.SAN(loss=criterion, regularizer=penalty, customized_dist=dist)
    if "SANA" in config.algorithms2run:
        algorithms2run["SANA"] = solvers.SANA(loss=criterion, regularizer=penalty)
    if "SANid" in config.algorithms2run:
        algorithms2run["SANid"] = solvers.SANid(loss=criterion, regularizer=penalty, customized_dist=dist)
    if "SAG" in config.algorithms2run:
        algorithms2run["SAG"] = solvers.SAG(loss=criterion, regularizer=penalty)
    if "SVRG" in config.algorithms2run:
        algorithms2run["SVRG"] = solvers.SVRG(loss=criterion, regularizer=penalty)
    if "ADAM" in config.algorithms2run:
        algorithms2run["ADAM"] = solvers.Adam(loss=criterion, regularizer=penalty,
                                              beta1=0.9, beta2=0.999, eps=1e-8)
    if "GD" in config.algorithms2run:
        algorithms2run["GD"] = solvers.GradientDescent(loss=criterion, regularizer=penalty)
    if "Newton" in config.algorithms2run:
        algorithms2run["Newton"] = solvers.Newton(loss=criterion, regularizer=penalty)

    kwargs = {"data": X, "label": y, "lr": config.lr, "reg": reg, "epoch": epochs,
              "x_0": x_0.copy(), "tol": config.tol, "verbose": config.verbose}
    for algo_name in algorithms2run.keys():
        solver_instance = algorithms2run[algo_name]
        np.random.seed(0)  # random seed to reproduce the experiments

        if "sag" in algo_name.lower() or "svrg" in algo_name.lower():
            if config.loss == "L2":
                lr = 1. / utils.max_Li_ridge(X, reg)
            elif config.loss == "Logistic":
                lr = 1. / utils.max_Li_logistic(X, reg)
            else:
                print("Warning!!!")
                lr = 0.01
            kwargs["lr"] = lr
        if "adam" in algo_name.lower():
            kwargs["lr"] = 0.001
        if "gd" in algo_name.lower():
            # 1/L, L is the smoothness constant
            if config.loss == "L2" and config.regularizer == "L2":
                lr = 1. / utils.lipschitz_ridge(X, reg)
            elif config.loss == "Logistic" and config.regularizer == "L2":
                lr = 1. / utils.lipschitz_logistic(X, reg)
            else:
                print("Warning!!! GD learning rate")
                lr = 0.01
            kwargs["lr"] = lr
        if "newton" in algo_name.lower():
            lr = 1.0
            kwargs["lr"] = lr

        grad_iter, grad_time, fval_iter = utils.run_algorithm(
            algo_name=algo_name.upper(), solver=solver_instance, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo_norm[algo_name.upper()] = grad_iter
        dict_algo_time[algo_name.upper()] = grad_time
        dict_algo_fval[algo_name.upper()] = fval_iter
        utils.save(os.path.join(folder_path, algo_name.lower() + '_grad_iter'), grad_iter,
                   os.path.join(folder_path, algo_name.lower() + '_grad_time'), grad_time,
                   os.path.join(folder_path, algo_name.lower() + '_fval_iter'), fval_iter)

    for algo_name in config.algorithms2load:
        grad_iter, grad_time, fval_iter = utils.load(os.path.join(folder_path, algo_name.lower() + '_grad_iter'),
                                                     os.path.join(folder_path, algo_name.lower() + '_grad_time'),
                                                     os.path.join(folder_path, algo_name.lower() + '_fval_iter'))
        if grad_iter:
            dict_algo_norm[algo_name.upper()] = grad_iter
        if grad_time:
            dict_algo_time[algo_name.upper()] = grad_time
        if fval_iter:
            dict_algo_fval[algo_name.upper()] = fval_iter

    # plot two figures: grad vs epoch and grad vs time
    if config.verbose:
        utils.plot_grad_iter(result_dict=dict_algo_norm, title=data_set, save_path=folder_path)
        utils.plot_grad_time(grad_dict=dict_algo_norm, time_dict=dict_algo_time,
                             title=data_set, save_path=folder_path)
        utils.plot_sub_optimality(result_dict=dict_algo_fval, title=data_set,
                                  save_path=folder_path, f_opt=f_opt)
