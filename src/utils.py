import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import tree
import graphviz


# Load the testbench data
def load_test():
    df = pd.read_pickle('data\\df_test.pkl')
    pivoted = df.pivot(index='sample_nr',columns='idx')
    X = np.stack([pivoted['position'].values, pivoted['velocity'].values, pivoted['current'].values], axis=2)
    y = df.groupby('sample_nr').target.first().values
    return X, y

# Load any dataset (WARNING: predefined length!)
def load_data(dataset):
    if dataset == 'test':
        X, y = load_test()
        sz = 230
    elif dataset == 'uc1':
        X, y = split_df(pd.read_pickle('data\\df_uc1.pkl'),
                        index_column='run_id',
                        feature_columns=['fldPosition', 'fldCurrent'],
                        target_name='target')
        # Length of timeseries for resampler and cnn
        sz = 38
    elif dataset == 'uc2':
        X, y = split_df(pd.read_pickle('data\\df_uc2.pkl'),
                        index_column='run_id',
                        feature_columns=['position', 'force'],
                        target_name='label')
        # Length of timeseries for resampler and cnn
        sz = 200
    resampler = TimeSeriesResampler(sz=sz)
    X = resampler.fit_transform(X, y)
    y = np.array(y)
    return X, y

# Load and split UC1 and UC2 datasets
def split_df(df,index_column, feature_columns, target_name):
    labels = []
    features = []
    for id_, group in df.groupby(index_column):
        features.append(group[feature_columns].values.tolist())
        labels.append(group[target_name].iloc[0])
    return features, labels

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    """

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    # Matplotlib 3.1.1 bug workaround
    ax.set_ylim(len(cm)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def to_time_series_dataset(dataset):
    """Transforms a time series dataset so that it has the following format:
    (no_time_series, no_time_samples, no_features)

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    Returns
    -------
    numpy.ndarray of shape
        (no_time_series, no_time_samples, no_features)
    """
    assert len(dataset) != 0, 'dataset is empty'

    try:
        np.array(dataset, dtype=np.float)
    except ValueError:
        raise AssertionError('All elements must have the same length.')

    if np.array(dataset[0]).ndim == 0:
        dataset = [dataset]

    if np.array(dataset[0]).ndim == 1:
        no_time_samples = len(dataset[0])
        no_features = 1
    else:
        no_time_samples, no_features = np.array(dataset[0]).shape

    return np.array(dataset, dtype=np.float).reshape(
        len(dataset),
        no_time_samples,
        no_features)


def to_dataset(dataset):
    """Transforms a time series dataset so that it has the following format:
    (no_time_series, no_time_samples, no_features) where no_time_samples
    for different time sereies can be different.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    Returns
    -------
    list of np.arrays
        (no_time_series, no_time_samples, no_features)
    """
    assert len(dataset) != 0, 'dataset is empty'

    if np.array(dataset[0]).ndim == 0:
        dataset = [[d] for d in dataset]

    if np.array(dataset[0]).ndim == 1:
        no_features = 1
        dataset = [[[d] for d in data] for data in dataset]
    else:
        no_features = len(dataset[0][0])

    for data in dataset:
        try:
            array = np.array(data, dtype=float)
        except ValueError:
            raise AssertionError(
                "All samples must have the same number of features!")
        assert array.shape[-1] == no_features,\
            'All series must have the same no features!'

    return dataset

class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    no_output_samples : int
        Size of the output time series.
    """
    def __init__(self, sz):
        self._sz = sz

    def fit(self, X, y=None, **kwargs):
        return self

    def _interp(self, x):
        return np.interp(
            np.linspace(0, 1, self._sz),
            np.linspace(0, 1, len(x)),
            x)

    def transform(self, X, **kwargs):
        X_ = to_dataset(X)
        res = [np.apply_along_axis(self._interp, 0, x) for x in X_]
        return to_time_series_dataset(res)

class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp.
    standard deviation) in each dimension. The mean and std can either be
    constant (one value per feature over all times) or time varying (one value
    per time step per feature).

    Parameters
    ----------
    kind: str (one of 'constant', or 'time-varying')
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.
    """
    def __init__(self, kind='constant', mu=0., std=1.):
        assert kind in ['time-varying', 'constant'],\
            'axis should be one of time-varying or constant'
        self._axis = (1, 0) if kind == 'constant' else 0
        self.mu_ = mu
        self.std_ = std

    def fit(self, X, y=None, **kwargs):
        X_ = to_time_series_dataset(X)
        self.mean_t = np.mean(X_, axis=self._axis)
        self.std_t = np.std(X_, axis=self._axis)
        self.std_t[self.std_t == 0.] = 1.

        return self

    def transform(self, X, **kwargs):
        """Fit to data, then transform it.
        Parameters
        ----------
        X
            Time series dataset to be rescaled
        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = to_time_series_dataset(X)
        X_ = (X_ - self.mean_t) * self.std_ / self.std_t + self.mu_

        return X_


class Flattener(TransformerMixin):
    """Flattener for time series. Reduces the dataset by one dimension by
    flattening the channels"""

    def __init__(self):
        pass

    def fit(self,X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """Transform data.
        Parameters
        ----------
        X
            Time series dataset to be rescaled
        Returns
        -------
        numpy.ndarray
            Flattened time series dataset
        """
        X_ = X.transpose(0, 2, 1).reshape(X.shape[0],-1)
        return X_

class Differentiator(TransformerMixin):
    """Calculates the derivative of a specified channel and and appends
    it as new channel"""
    def __init__(self, channel):
        """Initialise Featuriser.
        Parameters
        ----------
        channel
            int, channel to calculate derivative from
        """
        self.channel = channel

    def fit(self,X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """Transform data.
        Parameters
        ----------
        X
            Time series dataset
        Returns
        -------
        numpy.ndarray
            Time series dataset with new channel
        """
        dt = np.diff(X[:, :, self.channel], axis=1, prepend=X[0, 0, self.channel])
        X = np.concatenate((X, np.expand_dims(dt, axis=2)), axis=2)
        return X


class Featuriser(TransformerMixin, BaseEstimator):
    """Featuriser for time series. Calculates a set of statistical measures
     on each channel and each defined window of the dataset and returns a
     flattened matrix to train sklearn models on"""

    def __init__(self, windows=1):
        """Initialise Featuriser.
        Parameters
        ----------
        windows
            int, number of windows to part the time series in
        """
        self.windows = windows

    def fit(self,X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """Transform data.
        Parameters
        ----------
        X
            Time series dataset to be rescaled
        Returns
        -------
        numpy.ndarray
            Featurised time series dataset
        """
        X_ = np.empty((X.shape[0], 0))
        for i in range(X.shape[2]):
            for window in np.array_split(X[:, :, i], self.windows, axis=1):
                mean = np.mean(window, axis=1)
                std = np.std(window, axis=1)
                min_d = np.min(window, axis=1)
                min_loc = np.argmin(window, axis=1)
                max_d = np.max(window, axis=1)
                max_loc = np.argmax(window, axis=1)
                # Concatenate all values to a numpy array
                row = [mean, std, min_d, min_loc, max_d, max_loc]
                row = np.transpose(np.vstack(row))
                X_ = np.hstack([X_, row])
        return X_


class Featuriser2(TransformerMixin):
    """Deprecated. Featuriser for time series. Calculates a set of statistical measures
     on each channel of the dataset and returns a flattened matrix to train
     sklearn models on"""

    def __init__(self):
        pass

    def fit(self,X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """Transform data.
        Parameters
        ----------
        X
            Time series dataset to be rescaled
        Returns
        -------
        numpy.ndarray
            Featurised time series dataset
        """
        X_ = np.empty((X.shape[0], 0))
        for i in range(X.shape[2]):
            table = np.empty((0, 14))
            for x in X[:, :, i]:
                mean = np.mean(x)
                var = np.var(x)
                max_d = x.max()
                max_loc = np.argmax(x)
                min_d = x.min()
                min_loc = np.argmin(x)
                range_d = max_d - min_d
                med = np.median(x)
                first = x[0]
                last = x[-1]
                skew_d = skew(x)
                kurt = kurtosis(x)
                sum = np.sum(x)
                mean_abs_change = np.mean(np.abs(np.diff(x)))
                # Concatenate all values to a numpy array
                row = [mean, var, med, first, last, range_d, min_d, min_loc, max_d, max_loc, skew_d, kurt, sum,
                       mean_abs_change]
                row = np.hstack(row)
                table = np.vstack([table, row])
            X_ = np.hstack((X_,table))
        return X_

class Cutter(TransformerMixin):
    """Cuts the last part of the curves."""

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """Transform data.
        Parameters
        ----------
        X
            Time series dataset to be rescaled
        Returns
        -------
        list
            Cut time series dataset
        """
        res = []
        for x in X:
            idx = np.argmax(np.array(x)[:, 0])
            res.append(x[:idx])
        return res

def plot_dtc(dtc):
    feature_names = []
    #channels = ["$pos","$vel","$cur"] # test case
    #channels = ["$pos","$cur"] # use case 1
    #channels = ["$pos","$cur","$vel"] # use case 1 with derived velocity
    channels = ["$pos","$for"] # use case 2
    for var in channels:
        for i in range(1,int((dtc.n_features_/6/len(channels))+1)):
            for f in ["{mean}$","{std}$","{min}$","{min-ind}$","{max}$","{max-ind}$"]:
                feature_names.append('{0}^{1}_{2}'.format(var,i,f))
    
    #target_names = ["0","1","2","3","4"] # test case
    target_names = ["0","1","2","3"] # use case 1 + 2

    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=False, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'svg'
    graph.render("models\\dtc")
