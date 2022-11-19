## Rashomon Capacity
The official code of **Rashomon Capacity: A Metric for Predictive Multiplicity in Classification (NeurIPS 2022)** [[arXiv]](https://arxiv.org/abs/2206.01295)

#### Requirements
```
$ conda create --name <your_env_name> --file requirements.txt
$ source activate <your_env_name>
```

#### `data/` contains all datasets
- `UCI-Adult/`: raw data <ins> adult.data</ins> , <ins> adult.names</ins> , <ins> adult.test</ins>  [1].
- `COMPAS/`: raw data <ins> compas-scores-two-years.csv</ins>  [2]
- `HSLS/`: k-NN imputed HSLS dataset [3] ([Raw data and pre-processing](https://drive.google.com/drive/folders/14Ke1fiB5RKOVlA8iU9aarAeJF0g4SdBl))
```
./data
    /UCI-Adult/
        /adult.csv
        /adult.data
        /adult.names
        /adult.test
    /COMPAS/
        /compas-scores-two-years.csv
    /HSLS/
        /hsls_df_knn_impute_past_v2.pkl
    /cifar10/
```

#### `sampling/`: sampling method proposed in [4] to approximate the Rashomon set. 
- `adult-compas-hsls/`
  - command to run: *python3 <ins>sample-all.py</ins>*
- `cifar/`: 
  - command to run: *python3 <ins>sample-all.py</ins>*
- `utils/`: 
  - <ins>capacity.py</ins>: implementations of the Blahut-Arimoto (BA) algorithm to compute channel capacity.
  - <ins>training.py</ins>: 

#### `awp/`: Adversarial Weight Perturbation (AWP) method to explore the Rashomon set. 
- `adult-compas-hsls/`
  - command to run: *python3 <ins>perturb-all.py</ins>*
- `cifar/`: 
  - command to run: *python3 <ins>perturb-all.py</ins>*
- `utils/`: 
  - Python function <ins>load_data</ins> loads UCI-Adult and COMPAS datasets into PANDAS DataFrames.
  - Python function <ins>load_hsls_imputed</ins> loads the HSLS dataset into PANDAS DataFrames.
  - Python function <ins>load_cifar10</ins> loads CIFAR-10 [5] into the `data/cifar10/`.
  - Python function <ins>perturb_all_weights3</ins> performs AWP on multi-layer perceptrons (MLP) with UCI-Adult, COMPAS, and HSLS datasets.
  - Python function <ins>perturb_all_weights_cv3</ins> performs AWP on convolutional neural networks with the CIFAR-10 dataset.

#### Citation
```
@inproceedings{
  hsu2022rashomon,
  title={Rashomon Capacity: A Metric for Predictive Multiplicity in Classification},
  author={Hsiang Hsu and Flavio P. Calmon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://arxiv.org/abs/2206.01295}
}
```

#### Reference
[1] Lichman, M. (2013). UCI machine learning repository.

[2] Angwin, J., Larson, J., Mattu, S., and Kirchner, L. (2016). Machine bias. ProPublica.

[3] Ingels, S. J., Pratt, D. J., Herget, D. R., Burns, L. J., Dever, J. A., Ottem, R., Rogers, J. E., Jin, Y., and Leinwand, S. (2011). High school longitudinal study of 2009 (hsls: 09): Base-year data file documentation. nces 2011-328. National Center for Education Statistics.

[4] Semenova, L., Rudin, C., and Parr, R. (2019). A study in rashomon curves and volumes: A new perspective on generalization and model simplicity in machine learning. arXiv preprint arXiv:1908.01755.

[5] Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images (technical report). University of Toronto.
