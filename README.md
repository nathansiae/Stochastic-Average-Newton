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

2. Editing the `config.py` file.

	You need to edit some arguments to run the codes according to your experimental settings. 
	See the detailed explanations in this file.

3. Editing the `run.sh` file to specify the datasets you want to run.

3. Execute `chmod +x run.sh` in your terminal to make the script executable.

4. Running  `sh ./run.sh` in your terminal.



[sag]: https://arxiv.org/abs/1309.2388
[svrg]: https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
[snm]: https://arxiv.org/abs/1912.01597