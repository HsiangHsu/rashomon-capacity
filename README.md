# Rashomon Capacity
The official code of Rashomon Capacity: A Metric for Predictive Multiplicity in Probabilistic Classification (NeurIPS 2022) [[arxiv]](https://arxiv.org/abs/2206.01295)

## Quick Start
### 1. Requirements
```
$ conda create --name <your_env_name> --file requirements.txt
```

### 2. Prepare Datasets
1) Download datasets (UCI Adult[1], COMPAS[2], HSLS[3], CIFAR-10[4]) from [[this google drive link]](https://drive.google.com/file/d/19UaTcjGYj8YUBlj69mPK7zcVvFUR8bso/view?usp=sharing)
2) Locate downloaded datasets to './data' directory

```
./data
      /Permuted_Omniglot_task50.pt
      /binary_split_cub200_new
      /binary_split_cifar100
      /binary_cifar10
      /binary_omniglot
```

### 3.  Run .sh file
#### 3-1) Train 'CIFAR' scenarios using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_cifar.sh
```

#### 3-2) Train 'Omniglot' scenario using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_omniglot.sh
```

#### 3-3) Train 'CUB200' scenario using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_cub200.sh
```

### 3.  Analyze experimental results

1) Check './result_analysis_code/'. There are example ipython files to anayze the experimental results of [EWC, MAS, SI, Rwalk, AGS-CL] with or without CPR in CIFAR100. Note that the analysis results are for experiments conducted on only single seed.

2) You can easily transform and use these files to analyze other results!


## QnA
### 1. How to apply CPR to another CL algorithm?

: The implementation for CPR is quite simple. As shown in Equation (3) of the paper, you can implement CPR by maximizing an entropy of a model's softmax output (in other words, minimizing KL divergence between the model's softmax output and uniform distribution). Note that a lambda (the hyperparameter for entropy maximization) should be selected carefully. As an example, check Line 222 at './approaches/ewc_cpr.py'.


## Citation
```
@inproceedings{
  hsu2022rashomon,
  title={Rashomon Capacity: A Metric for Predictive Multiplicity in Probabilistic Classification},
  author={Hsiang Hsu and Flavio P. Calmon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://arxiv.org/abs/2206.01295}
}
```

## Reference
[1] Lichman, M. (2013). UCI machine learning repository.
[2] Angwin, J., Larson, J., Mattu, S., and Kirchner, L. (2016). Machine bias. ProPublica.
[3] Ingels, S. J., Pratt, D. J., Herget, D. R., Burns, L. J., Dever, J. A., Ottem, R., Rogers, J. E., Jin, Y., and Leinwand, S. (2011). High school longitudinal study of 2009 (hsls: 09): Base-year data file documentation. nces 2011-328. National Center for Education Statistics.
[4] Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images (technical report). University of Toronto.
