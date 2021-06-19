import os
import pickle
import loss
import regularizer
import numpy as np
from sklearn.model_selection import train_test_split
import load_data
from algorithms import san, sag, svrg
import utils

if __name__ == "__main__":
    #############################################
    # set experimental parameters ###############
    # for simplicity, we don't use here parser ##
    problem_type = "classification"  # regression
    data_name = "phishing"
    data_path = "./datasets/phishing.txt"
    save_path = os.path.join("./grid_search", data_name)
    algo_run = sag  # san, sag, svrg
    algo_name = "sag"
    loss_type = "Logistic"  # L2, PseudoHuber
    reg_type = "L2"  # PseudoHuber
    max_epochs = 50
    repetition = 5
    threshold = 1e-4
    # end of setting ############################
    #############################################

    np.random.seed(0)

    X, y = load_data.get_data(data_path)
    X = X.toarray()  # convert from scipy sparse matrix to dense
    print("Data Sparsity: {}".format(load_data.sparsity(X)))

    n, d = X.shape
    if n > 20000:  # if the dataset is too large, we do grid search on a small subset to save the time
        if problem_type == "classification":
            if d > 1000:
                X, _, y, _ = train_test_split(X, y, train_size=5000,
                                              random_state=np.random.RandomState(0), stratify=y)
            else:
                X, _, y, _ = train_test_split(X, y, train_size=10000,
                                              random_state=np.random.RandomState(0), stratify=y)
        elif problem_type == "regression":  # for regression dataset, we don't use stratify to select the subset
            if d > 1000:
                X, _, y, _ = train_test_split(
                    X, y, train_size=5000, random_state=np.random.RandomState(0))
            else:
                X, _, y, _ = train_test_split(X, y, train_size=10000,
                                              random_state=np.random.RandomState(0))
        else:
            raise Exception("Unknown problem type!")

    if problem_type == "classification":
        # label preprocessing for some dataset whose label is not {-1, 1}
        max_v, min_v = np.max(y), np.min(y)
        idx_min = (y == min_v)
        idx_max = (y == max_v)
        y[idx_min] = -1
        y[idx_max] = 1
    elif problem_type == "regression":
        # X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)  # scale features to mean=0, var=1
        X = X / np.max(np.abs(X), axis=0)  # scale features to [-1, 1]
    else:
        raise Exception("Unknown problem type!")
    # adding a column filling with all ones.(bias term)
    X = np.c_[X, np.ones(X.shape[0])]

    if loss_type == "L2":
        criterion = loss.L2()
    elif loss_type == "PseudoHuber":
        criterion = loss.PseudoHuberLoss(delta=1.0)
    elif loss_type == "Logistic":
        criterion = loss.LogisticLoss()
    else:
        raise Exception("Unknown loss function!")

    if reg_type == "L2":
        penalty = regularizer.L2()
    elif reg_type == "PseudoHuber":
        penalty = regularizer.PseudoHuber(delta=1.0)
    else:
        raise Exception("Unknown regularizer type!")

    n, d = X.shape
    reg = 1. / n
    x_0 = np.zeros(d)

    if loss_type == "L2" and reg_type == "L2":
        lr_base = 1. / utils.max_Li_ridge(X, reg)
    elif loss_type == "Logistic" and reg_type == "L2":
        lr_base = 1. / utils.max_Li_logistic(X, reg)
    else:
        print("Warning!!! No theory learning rate")
        lr_base = 0.01

    records = []
    if algo_name == "san":
        lr_candidates = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        p_0_candiates = [0.5/n, 1./n, 10./n, 100./n, 1000./n]
        for p_0 in p_0_candiates:
            print(p_0)
            dist = np.array([(1 - p_0) / n] * n + [p_0])
            r = []
            for j, lr in enumerate(lr_candidates):
                data_pass = 0
                for _ in range(repetition):
                    final_w, norm, time = algo_run(loss=criterion, regularizer=penalty,
                                                   data=X, label=y, lr=lr, reg=reg,
                                                   epoch=max_epochs, x_0=x_0.copy(),
                                                   tol=threshold, verbose=0, dist=dist)
                    data_pass += len(norm) - 1
                # accuracy
                y_hat = np.sign(X @ final_w)
                print("accuracy: {:f}".format(np.sum(y_hat == y)/n))
                r.append(data_pass // repetition)
                print("lr: {}; iterations: {}".format(lr, r[-1]))
            records.append(r.copy())
            print_str = ""
            for num in r:
                print_str += "${}$ & ".format(num)
            print(print_str)
    elif algo_name == "sag" or algo_name == "svrg":
        lr_candidates = [lr_base / 10., lr_base / 5., lr_base /
                         3., lr_base / 2., lr_base, 2*lr_base, 5*lr_base]
        for j, lr in enumerate(lr_candidates):
            data_pass = 0
            for _ in range(repetition):
                final_w, norm, time = algo_run(loss=criterion, regularizer=penalty,
                                               data=X, label=y, lr=lr, reg=reg,
                                               epoch=max_epochs, x_0=x_0.copy(),
                                               tol=threshold, verbose=0)
                data_pass += len(norm) - 1
            # accuracy
            y_hat = np.sign(X @ final_w)
            print("accuracy: {:f}".format(np.sum(y_hat == y)/n))
            records.append(data_pass // repetition)
            print("lr: {}; iterations: {}".format(lr, records[-1]))
        print_str = ""
        for num in records:
            print_str += "${}$ & ".format(num)
        print(print_str)
    else:
        raise Exception("Unknown algorithm name !")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, algo_name), 'wb') as fp:
        pickle.dump(records, fp)
