# ALPACA
**A** **L**earning **P**roduct **A**ssembly **C**lassification **A**lgorithm

Timeseries classification algorithm aiming to avoid False Negatives in the classification of process signals. This code complements the paper submitted to the journal of [Decision Support Systems](https://www.journals.elsevier.com/decision-support-systems) _"Applied Machine Learning for a Zero Defect Tolerance System in the Automated Assembly of Pharmaceutical Devices"_ by Dengler _et. al._ 

Alpaca automatically performes a gridsearch cross-validation for each of the algorithms elements and composes the final classifier of the best performing individual models. For the cross-validation the number of folds _k_ is equal to 5. For each model a different hyper-parameter space is searched. The following tables show the hyperparameter that are included in the gridsearch and the range for each algorithm in the ensemble.

**Anomaly Detection**
| Parameter          | Range         |
|--------------------|---------------|
| Number of clusters | 10,50,100,200 |

**Decision Tree**
| Parameter          | Range             |
|--------------------|-------------------|
| Number of windows  | 1, 2, 3, 4, 5, 6  |
| Maximum depth      | 3, 4, 5           |
| Criterion          | 'gini', 'entropy' |

**Support Vector Machines**
| Parameter                             | Range                                     |
|---------------------------------------|-------------------------------------------|
| C                                     | 10, 100, 1000, 10000                      |                     
| γ (in case of rbf kernel)             | 0.01, 0.001, 0.0001, 0.00001              |
| Degree (in case of polynomial kernel) | 2, 3                                      |
| Kernel                                | Radial Basis Function, Linear, Polynomial |

**Convolutional Neural Network**
| Parameter                                                | Range                        |
|----------------------------------------------------------|------------------------------|
| Batch Size                                               | 20, 30                       |
| Number of convolutional layers                           | 1, 2                         |
| Number of filters                                        | 4, 8, 16                     |
| Length of the filters (as fraction of the signal length) | 0.05, 0.1, 0.2               |
| Activation function                                      | ReLU                         |
| Optimizer                                                | ADAM                         |
| Loss function                                            | Categorical cross-entropy    |

## 1. Requirements

The sourcecode of this project was written in Python 3.7.4 and has dependencies in the following packages:
```
numpy
scipy
pandas
graphviz
ipython
cython
scikit-learn==0.22.1
tslearn==0.2.5
tensorflow==1.15.4
```

Install the dependencies via the following command:
```
pip install -r /path/to/requirements.txt
```
Some of the used packages might have dependencies outside of Python.

## 2. Executing the program

The algorithm adapts methods similar to the standards of the `scikit-learn` library.

A pipeline can be set up such as:
```
alpaca = Pipeline([('resampler', TimeSeriesResampler(sz=sz)),('alpaca', Alpaca())])
```

and can be trained by calling:
```
alpaca.fit(X_train, y_train)
```

After training predictions can be made with:
```
y_pred_bin, y_pred = alpaca.predict(X_test, voting="veto")
```
Notice that unlike the `predict` method known from `scikit-learn` this implementation takes an additional argument `voting` and returns two arrays as predictions. 
The voting argument can be varied between `democratic` and `veto` to adjust the voting scheme of the ensemble. Stacked classifiers can be utilized, too, by assigning `voting` to `meta_dtc` or `meta_svc`. 

The first return argument (`y_pred_bin`) contains the quality result and only differentiates between rejected and accepted. The second returned array (`y_pred`) includes the class predictions of the calssifier ensemble.

The pipeline accepts data `X` in form of a 3D numpy array or a list of lists. In case of a numpy array the dimensions are: `X[n_samples,n_points_per_timeseries,n_channels]`. In case of a list of lists, the structure is similar: `X[n_samples][n_points_per_timeseries][n_channels]`.
 The advantage of using a list of lists is, that the timeseries do not have to be the same length troughout every sample, and can be easily resampled by using a `TimeSeriesResampler()` in the pipeline.

The label `y` can either be a 1D numpy array or a list of class values. Number 0 must be assigned to the negative class. The different positive classes are assigned the subsequent values (1,2,3,...).

Two examples of possible pipelines are given in the `main.py` file.