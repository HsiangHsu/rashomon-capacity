# Rashomon Capacity
The official code of Rashomon Capacity: A Metric for Predictive Multiplicity in Probabilistic Classification (NeurIPS 2022) [[arxiv]](https://arxiv.org/abs/2206.01295)

## Quick Start
### 1. Requirements
```
$ conda create --name <your_env_name> --file requirements.txt
```

### 2. Prepare Datasets
1) The './data' directory contains UCI Adult[1], COMPAS[2], and HSLS[3] (missing values imputed by knn) datasets. 
2) Locate downloaded datasets to './data' directory CIFAR-10[4]

```
./data
    /UCI-Adult
        /adult.csv
        /adult.data
        /adult.names
        /adult.test
    /COMPAS
        /compas-scores-two-years.csv
    /HSLS
        /hsls_df_knn_impute_past_v2.pkl
    /cifar10
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
