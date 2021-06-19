#!/bin/sh

# This shell script can test multiple datasets by one launch.
# You need to put all the datasets names you want to run in the array DATASETS_NAMES, separated by space;
# You also need to put all corresponding paths to load the dataset in the array DATASETS_PATHS, with the same
# order as DATASETS_NAMES(idem. separated also by space).
# The others arguments can be set in for loop but notice that these arguments keep the same for all datasets to run

# ==========================================================================
# Explanation of arguments:
# --type: int, type of problem, 0 means classification and 1 means regression
# --dataset: str, name of dataset.
# --data_path: str, path to load dataset
# --result_folder: str, name of folder to store the results
# --log_file: str, name of log file to record experiments details
# --epochs: int, epochs to run for one algorithm, i.e., effective data passes
# --n_repetitions: int, number of times to repeat for one algorithm
# --reg: float, regularization coefficient, i.e., \lambda in paper. default: None
# --ill_conditional: int, 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2; this argument will be ignored if reg is set
# --lr: float, learning rate for SAN/SANA. default: 1.0
# --tol: float, the algorithm will be cut when the norm of gradient reaches below this threshold. default: None
# --loss: str, support "Logistic" for classification; "L2" or "PseudoHuber" for regression. default: "L2"
# --regularizer: str, support "L2" or "PseudoHuber". default: "L2"
# --run_xx: bool, do we run xx algorithm
# ==========================================================================

DATASETS_NAMES=("covtype" "phishing")
DATASETS_PATHS=("./datasets/covtype" "./datasets/phishing.txt")
NUM_DATASETS=${#DATASETS_NAMES[@]}
for (( i=0; i<$NUM_DATASETS; i++ ))
do
  echo "START Running ${DATASETS_NAMES[i]}"
  python main.py --type 0 --dataset ${DATASETS_NAMES[i]} --data_path ${DATASETS_PATHS[i]} \
                 --result_folder 'TBD' --log_file 'log.txt' \
                 --epochs 50 --n_repetition 10 --ill_conditional 2 --lr 1.0 \
                 --loss "Logistic" --regularizer 'L2'  \
                 --run_san True --run_sag True --run_svrg True --run_san_id False \
                 --run_sana False --run_snm False --run_gd False --run_newton False
  echo "Finished ${DATASETS_NAMES[i]}"
done
