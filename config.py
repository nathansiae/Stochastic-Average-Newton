"""
All experiments settings should be controlled in this file
"""
from algorithms import san, sag, svrg, snm, sana, gd, newton, san_id, iqn

problem_type = 'classification'  # type of problem, supports 'classification' and 'regression'.

output_path = 'new_branch'  # folder path to store experiments results

log_file = 'log.txt'  # log file name

n_repetition = 3  # number of repetitions run for stochastic algorithm

epochs = 20  # number of epochs to run for one algorithm, i.e., effective data passes

ill_conditional = 1  # can be chosen in 1,2,3. 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2. this argument will be ignored if reg is set

loss = 'Logistic'  # str, loss function name; support "Logistic" for classification; "L2" or "PseudoHuber" for regression

regularizer = "L2"  # str, regularizer type name; support "L2" or "PseudoHuber"

reg = None  # float, regularization coefficient, i.e., \lambda in paper. default: None

lr = 1.0  # float, learning rate for SAN/SANA. default: 1.0

tol = 1e-8  # float, the algorithm will be cut when the norm of gradient reaches below this threshold.

verbose = 1  # Should we save the outputs (data and plots)? 1 for yes, 0 for no. Default is 1.

subopt = 0  # Should we plot sub-optimality curves? 1 for yes, 0 for no. Default is 0.

is_uniform = True  # for some algorithms, we may choose non-uniform sampling technique

algorithms2run = {"SAN": san, "SAG": sag, "SVRG": svrg}  # dictionary, all algorithm to test

algorithms2load = []  
# list of algorithm names whose results are ready to load(make sure the results exist).
# for example, under output_path folder, we have results named "san_grad_iter", and we want load it directly,
# then we can put san in the list.
