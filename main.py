import os
import argparse
import logging
import math
import time
import numpy as np
import load_data
from algorithms import san, sag, svrg, snm, vsn, sana, svrg2, gd, newton
import utils
import loss
import regularizer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', action='store', dest='type', type=int, default=0,
                        help="type of problem, 0 means classification and 1 means regression.")
    parser.add_argument('--dataset', action='store', dest='data_set',
                        help="data set name")
    parser.add_argument('--data_path', action='store', dest='data_path',
                        help='path to load data')
    parser.add_argument('--result_folder', action='store', dest='folder',
                        help="folder path to store experiments results")
    parser.add_argument('--log_file', default='log.txt')
    parser.add_argument('--n_repetition', action='store', type=int, dest='n_rounds', default=10,
                        help="number of repetitions run for algorithm")
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=100)
    parser.add_argument('--ill_conditional', action='store', dest='ill_conditional', type=int, default=2,
                        help="can be chosen in 1,2,3. 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.")
    parser.add_argument('--loss', default="L2", help="loss function")
    parser.add_argument('--regularizer', default="L2", help="regularizer type")
    parser.add_argument('--reg', action='store', type=float, dest='reg', default=None)
    parser.add_argument("--lr", action='store', type=float, dest='lr', default=1.0)
    parser.add_argument('--run_san', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sana', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sag', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg2', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_snm', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_vsn', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_gd', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_newton', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    opt = parser.parse_args()

    np.random.seed(0)  # random seed to reproduce the experiments

    data_set = opt.data_set
    folder_path = os.path.join(opt.folder, data_set)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.basicConfig(filename=os.path.join(folder_path, opt.log_file), level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    logging.info(opt)

    is_uniform = True  # do we use uniform sampling for SAN?

    # load data
    if data_set == "artificial":
        d, n = 30, 5000
        idx = np.arange(d)
        # Ground truth coefficients of the model
        x_model_truth = (-1) ** idx * np.exp(-idx / 10.)  # np.random.randn(d)  #
        if opt.type == 0:
            X, y = load_data.simu_logreg(x_model_truth, n, corr=0.1)
        elif opt.type == 1:
            X, y = load_data.simu_linreg(x_model_truth, n, corr=0.1)
        else:
            raise Exception("Unknown problem type!")
        # X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    else:
        X, y = load_data.get_data(opt.data_path)
        X = X.toarray()  # convert from scipy sparse matrix to dense
        # adding a column filling with all ones.(bias term)
        X = np.c_[X, np.ones(X.shape[0])]
        if opt.type == 0:
            # X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
            # label preprocessing for some dataset whose label is not {-1, 1}
            max_v, min_v = np.max(y), np.min(y)
            y[y == max_v] = 1
            y[y == min_v] = -1
        elif opt.type == 1:
            # X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)  # scale features to mean=0, var=1
            X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
        else:
            raise Exception("Unknown problem type!")

    n, d = X.shape
    logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))

    if opt.loss == "L2":
        criterion = loss.L2()
    elif opt.loss == "PseudoHuber":
        criterion = loss.PseudoHuberLoss(delta=1.0)
    elif opt.loss == "Logistic":
        criterion = loss.LogisticLoss()
    else:
        raise Exception("Unknown loss function!")

    if opt.regularizer == "L2":
        penalty = regularizer.L2()
    elif opt.regularizer == "PseudoHuber":
        penalty = regularizer.PseudoHuber(delta=1.0)
    else:
        raise Exception("Unknown regularizer type!")

    if opt.reg is None:
        if opt.ill_conditional == 3:
            reg = 1. / (n * n)  # ill conditional
        elif opt.ill_conditional == 1:
            reg = 1. / math.sqrt(n)
        else:
            reg = 1. / n
    else:
        reg = opt.reg
    logging.info("Regularization param: {}".format(reg))

    epochs = opt.epochs
    n_rounds = opt.n_rounds
    x_0 = np.zeros(d)  # np.random.randn(d)

    if is_uniform:
        dist = None
    else:
        p_0 = 1. / (n + 1)
        logging.info("Probability p_0: {:}".format(p_0))
        dist = np.array([p_0] + [(1 - p_0) / n] * n)

    dict_algo = {}

    if opt.run_svrg2:
        svrg2_lr = 0.001*reg
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg2_lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SVRG2", algo=svrg2, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SVRG2"] = grad_iter
        utils.save(os.path.join(folder_path, 'svrg2_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'svrg2_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'svrg2_grad_iter'),
                                          os.path.join(folder_path, 'svrg2_grad_time'))
        if grad_iter:
            dict_algo["SVRG2"] = grad_iter

    if opt.run_san:
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": opt.lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SAN", algo=san, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SAN"] = grad_iter
        utils.save(os.path.join(folder_path, 'san_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'san_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'san_grad_iter'),
                                          os.path.join(folder_path, 'san_grad_time'))
        if grad_iter:
            dict_algo["SAN"] = grad_iter

    if opt.run_sana:
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SANA", algo=sana,
                                                   algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SANA"] = grad_iter
        utils.save(os.path.join(folder_path, 'sana_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sana_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'sana_grad_iter'),
                                          os.path.join(folder_path, 'sana_grad_time'))
        if grad_iter:
            dict_algo["SANA"] = grad_iter

    if opt.run_vsn:
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs, "x_0": x_0.copy()}
        grad_iter, grad_time = utils.run_algorithm(algo_name="VSN", algo=vsn, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["VSN"] = grad_iter
        utils.save(os.path.join(folder_path, 'vsn_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'vsn_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'vsn_grad_iter'),
                                          os.path.join(folder_path, 'vsn_grad_time'))
        if grad_iter:
            dict_algo["VSN"] = grad_iter

    if opt.run_sag:
        if opt.loss == "L2" and opt.regularizer == "L2":
            sag_lr = 1. / utils.max_Li_ridge(X, reg)
        elif opt.loss == "Logistic" and opt.regularizer == "L2":
            sag_lr = 1. / utils.max_Li_logistic(X, reg)
        else:
            print("Warning!!! SAG learning rate")
            sag_lr = 0.01
        # in the SAG paper, the lr given by theory is 1/16L.
        # sag_lr = 0.25 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SAG: {:f}".format(sag_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sag_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SAG", algo=sag, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SAG"] = grad_iter
        utils.save(os.path.join(folder_path, 'sag_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sag_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'sag_grad_iter'),
                                          os.path.join(folder_path, 'sag_grad_time'))
        if grad_iter:
            dict_algo["SAG"] = grad_iter

    if opt.run_svrg:
        if opt.loss == "L2" and opt.regularizer == "L2":
            svrg_lr = 1. / utils.max_Li_ridge(X, reg)
        elif opt.loss == "Logistic" and opt.regularizer == "L2":
            svrg_lr = 1. / utils.max_Li_logistic(X, reg)
        else:
            print("Warning!!! SVRG learning rate")
            svrg_lr = 0.01
        # in the book "Convex Optimization: Algorithms and Complexity, Sébastien Bubeck",
        # the Theorem 6.5 indicates that the theory choice lr of SVRG should be 1/10L.
        # svrg_lr = 0.4 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SVRG: {:f}".format(svrg_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SVRG", algo=svrg, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SVRG"] = grad_iter
        utils.save(os.path.join(folder_path, 'svrg_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'svrg_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'svrg_grad_iter'),
                                          os.path.join(folder_path, 'svrg_grad_time'))
        if grad_iter:
            dict_algo["SVRG"] = grad_iter

    if opt.run_snm:
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs, "x_0": x_0.copy()}
        grad_iter, grad_time = utils.run_algorithm(algo_name="SNM", algo=snm, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_algo["SNM"] = grad_iter
        utils.save(os.path.join(folder_path, 'snm_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'snm_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'snm_grad_iter'),
                                          os.path.join(folder_path, 'snm_grad_time'))
        if grad_iter:
            dict_algo["SNM"] = grad_iter

    if opt.run_gd:
        # 1/L, L is the smoothness constant
        if opt.loss == "L2" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_ridge(X, reg)
        elif opt.loss == "Logistic" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_logistic(X, reg)
        else:
            print("Warning!!! GD learning rate")
            gd_lr = 0.01
        logging.info("Learning rate used for Gradient descent: {:f}".format(gd_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": gd_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="GD", algo=gd, algo_kwargs=kwargs, n_repeat=1)
        dict_algo["GD"] = grad_iter
        utils.save(os.path.join(folder_path, 'gd_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'gd_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'gd_grad_iter'),
                                          os.path.join(folder_path, 'gd_grad_time'))
        if grad_iter:
            dict_algo["GD"] = grad_iter

    if opt.run_newton:
        newton_lr = 1.0
        logging.info("Learning rate used for Newton method: {:f}".format(newton_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": newton_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty}
        grad_iter, grad_time = utils.run_algorithm(algo_name="Newton", algo=newton, algo_kwargs=kwargs, n_repeat=1)
        dict_algo["Newton"] = grad_iter
        utils.save(os.path.join(folder_path, 'newton_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'newton_grad_time'), grad_time)
    else:
        grad_iter, grad_time = utils.load(os.path.join(folder_path, 'newton_grad_iter'),
                                          os.path.join(folder_path, 'newton_grad_time'))
        if grad_iter:
            dict_algo["Newton"] = grad_iter

    utils.plot_grad_iter(result_dict=dict_algo, title=data_set, save_path=folder_path)
