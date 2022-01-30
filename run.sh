#!/bin/sh
# This shell script can test multiple datasets by one launch.
# You need to put all the datasets names you want to run in the array DATASETS_NAMES, separated by space;
# You also need to put all corresponding paths to load the dataset in the array DATASETS_PATHS, with the same
# order as DATASETS_NAMES(idem. separated also by space).
# The others arguments can be set in for loop but notice that these arguments keep the same for all datasets to run

DATASETS_NAMES=("fourclass")
DATASETS_PATHS=("./datasets/fourclass.txt")

NUM_DATASETS=${#DATASETS_NAMES[@]}
for (( i=0; i<$NUM_DATASETS; i++ )) do
  echo "START Running ${DATASETS_NAMES[i]}"
  python main.py --dataset ${DATASETS_NAMES[i]} --data_path ${DATASETS_PATHS[i]}
  echo "Finished ${DATASETS_NAMES[i]}"
done
