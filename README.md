# Stochastic Average Newton methods
---

We provide here the implementations of two new stochastic Newton-type algorithms to solve the finite-sum minimization problem in Machine learning, *Stochastic Average Newton Method (SAN)* and *Stochastic Average Newton Alternative (SANA)*. Currently our implementations support only generalized linear models (GLMs). For loss function, logistic loss is provided for binary classification problems, pseudo-Huber loss and L2 loss are supported for regression problems. To compare our algorithms with the state-of-the-art algorithms for solving GLMs, we provide our codes for [SAG][sag] and [SVRG][svrg]. We also provide our codes for [Stochastic Newton][snm] as a bench mark for second order incremental algorithms.


## Package Requirements
---

+ numpy >= 1.13
+ matplotlib >= 2.1
+ scikit-learn >= 0.19


## Usage
---

1. Preparing dataset.

	The datasets we used in our experiments are downloaded from [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). They were put in the folder './datasets/'.

2. Editing the `run.sh` file.

	You need to edit some arguments to run the codes according to your experimental setting. For example,
	```
	python main.py --type 0 --dataset 'phishing' --data_path './datasets/phishing.txt' \
               	   --result_folder 'results' --log_file 'log.txt' \
                   --epochs 10 --n_repetition 2 --ill_conditional 2 --lr 1.0 \
                   --run_rows True --run_sag True --run_svrg True --run_snm True --run_vsn True
	```

	Explanation of arguments:

	+  --type: int, type of problem, 0 means classification and 1 means regression
    +  --dataset: str, name of dataset
 	+  --data_path: str, path to load dataset
 	+  --result_folder: str, name of folder to store the experimental results
 	+  --log_file: str, name of log file
 	+  --epochs: int, epochs to run for one algorithm
 	+  --n_repetitions: int, number of times to repeat for one algorithm
 	+  --ill_conditional: int, 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.
 	+  --reg: float, regularization parameter, default-None. If you set this argument, then ill_conditional will be ignored automatically.
 	+  --lr: float, learning rate for Rows Splitting Newton, default: 1.0
 	+  --run_xx, boolean, set True if you want to run *xx* algorithm

3. Execute `chmod +x run.sh` in your terminal to make the script executable.

4. Running  `./run.sh` in your terminal.



[sag]: https://arxiv.org/abs/1309.2388
[svrg]: https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
[snm]: https://arxiv.org/abs/1912.01597