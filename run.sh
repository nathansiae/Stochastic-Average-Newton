#!/bin/sh

# --type: int, type of problem, 0 means classification and 1 means regression
# --dataset: str, name of dataset. hint: can be set "artificial"
# --data_path: str, path to load dataset
# --result_folder: str, name of folder to store the results
# -- epochs: int, epochs to run for one algorithm
# --n_repetitions: int, number of times to repeat for one algorithm
# --ill_conditional: int, 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.
# --loss: str, support "Logistic" for classification; "L2" or "PseudoHuber" for regression. default: "L2"
# --regularizer: str, support "L2" or "PseudoHuber". default: "L2"
# --reg: float, regularization parameter, default-None
# --lr: float, learning rate for Stochastic Averaging Newton, default: 1.0
# --run_xx: do we run xx algorithm

python main.py --type 1 --dataset 'space-ga' --data_path './datasets/space_ga.txt' \
               --result_folder 'ridge' --log_file 'log.txt' \
               --epochs 50 --n_repetition 10 --ill_conditional 2 \
               --loss "L2" --regularizer 'L2'  \
               --run_san True --run_sag True --run_svrg True --run_svrg2 False --run_newton False --run_gd False