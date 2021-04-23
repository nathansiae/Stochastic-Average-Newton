# Stochastic-Variance-Reduced-Newton-Methods
---

We provide here the implementations of two stochastic Newton-type algorithms to solve the finite-sum minimization problem in Machine learning, *Row Sampled Newton Method(ROWS)* and *Variable Splitting Newton(VSN)*. Currently our implementations support only (generalized)linear models. For loss function, logistic loss is provided for binary classification problems and pseudo-huber loss is supported for robust regression problems. To compare our algorithms with the SOTA, we also provides our codes for [SAG][sag], [SVRG][svrg] and [Stochastic Newton][snm].


## Package Requirements
---

+ numpy >= 1.13
+ matplotlib >= 2.1
+ scikit-learn >= 0.19


## Usage
---

1. Preparing dataset.

	The datasets we used in our experiments are downloaded from [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). 

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


3. Running  `./run.sh` in your terminal.



[sag]: https://arxiv.org/abs/1309.2388
[svrg]: https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
[snm]: https://arxiv.org/abs/1912.01597
