import warnings
warnings.simplefilter(action='ignore')
import pickle
import pandas as pd
import numpy as np

from utils import TimeSeriesScalerMeanVariance, Flattener, Featuriser, plot_dtc

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin, BaseEstimator, clone

from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model


class Alpaca(ClassifierMixin):
    """
    A learning product classification algorithm.
    """
    def __init__(self):
        self.anomaly_detection = AnomalyDetection()
        self.classifier = Classifier()

    def fit(self, X, y, stacked=True):
        """
        Fit the algorithm according to the given training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Training samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        stacked:  bool
            If true train a meta classifier on kfold CV predictions of the level 1 classifiers
        Returns
        -------
        self: object
            Fitted model
        """
        # Fit anomaly detection
        # Do GridSearch to get best model
        param_grid = {'n_clusters': [10,50,100,200]}
        grid = GridSearchCV(self.anomaly_detection, param_grid, cv=5, refit=True, verbose=2, n_jobs=-1)
        grid.fit(X, y)
        
        # Save results
        df_results = pd.DataFrame.from_dict(data=grid.cv_results_)
        df_results.to_csv("results\\ad.csv",index=False)
        print(grid.best_params_)
        # Take best model
        self.anomaly_detection = grid.best_estimator_
        # Save the model
        with open("models\\ad.pkl", 'wb') as file:
            pickle.dump(self.anomaly_detection, file)

        # Fit ensemble classifier
        self.classifier.fit(X, y, stacked)

        return self

    def predict(self, X, voting):
        """
        Perform a classification on samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Test samples.
        voting: string
            Voting scheme to use
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Predictions from ensemble with suggested class labels
        y_pred_bin: array, shape (n_samples,)
            Combined binary predictions
        """
        # Class predictions of ensemble
        y_pred, y_pred_ens = self.classifier.predict(X, voting=voting)
        # Binary predictions of anomaly detector
        y_pred_ad = self.anomaly_detection.predict(X)
        # Save individual predictions
        y_pred_indiv = np.column_stack((y_pred_ens, y_pred_ad)).astype(int)
        df_results = pd.DataFrame(y_pred_indiv, columns = ['y_pred_dtc','y_pred_svc','y_pred_cnn','y_pred_ad'])
        df_results.to_csv("results\\y_pred_indiv.csv",index=False)

        # Overwrite the entries in y_pred_knn with positive, where ensemble decides positive
        y_pred_bin = np.where(y_pred != 0, 1, y_pred_ad)
        return y_pred_bin, y_pred


class AnomalyDetection(ClassifierMixin, BaseEstimator):
    """
    Anomaly detection with 1-NN and automatic calculation of optimal threshold.
    """
    def __init__(self, n_clusters=200):
        self.knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, weights='uniform', metric='euclidean', n_jobs=-1)
        self.d = None
        self.n_clusters = n_clusters

    def fit(self, X, y):
        """
        Fit the algorithm according to the given training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Training samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        Returns
        -------
        self: object
            Fitted model
        """
        # Fit anomaly detection knn over k-means centroids
        X_good = X[np.where(y == 0)]
        X_bad = X[np.where(y != 0)]
        km = TimeSeriesKMeans(n_clusters=self.n_clusters, metric="euclidean",
                              max_iter=100, random_state=0, n_jobs=-1).fit(X_good)
        self.knn.fit(km.cluster_centers_, np.zeros((self.n_clusters,)))

        # Calculate distances to all samples in good and bad
        d_bad, _ = self.knn.kneighbors(X_bad)
        d_good, _ = self.knn.kneighbors(X_good)

        # Calculate ROC
        y_true = np.hstack((np.zeros(X_good.shape[0]), np.ones(X_bad.shape[0])))
        y_score = np.vstack((d_good, d_bad))
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

        # Determine d by Youden index
        self.d = thresholds[np.argmax(tpr - fpr)]
        return self

    def predict(self, X):
        """
        Perform a classification on samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Test samples.
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Predictions
        """
        # Binary predictions of anomaly detector
        y_pred = np.squeeze(np.where(self.knn.kneighbors(X)[0] < self.d, 0, 1))
        return y_pred


class Classifier(ClassifierMixin):
    """
    Classifier part with ensemble of estimators.
    """
    def __init__(self):

        # DTC pipeline
        featuriser = Featuriser()
        dtc = DecisionTreeClassifier()
        self.dtc_pipe = Pipeline([('featuriser', featuriser), ('dtc', dtc)])
 
        # SVC pipeline
        scaler = TimeSeriesScalerMeanVariance(kind='constant')
        flattener = Flattener()
        svc = SVC()
        self.svc_pipe = Pipeline([('scaler', scaler), ('flattener', flattener), ('svc', svc)])

        # Keras pipeline
        #len_filter = round(len_input*0.05)
        #num_filter = 8
        cnn = KerasClassifier(build_fn=build_cnn, epochs=100, verbose=0)
        self.cnn_pipe = Pipeline([('scaler', scaler), ('cnn', cnn)])
        
        # Meta classifier
        self.meta_dtc = DecisionTreeClassifier()
        self.meta_svc = SVC()

    def fit(self, X, y, stacked):
        """
        Fit each individual estimator of the ensemble model according to the given training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Training samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        stacked:  bool
            If true train a meta classifier on kfold CV predictions of the level 1 classifiers
        Returns
        -------
        self: object
            Fitted model
        """
        # Fit DTC
        # Do GridSearch to get best model
        param_grid = {'featuriser__windows': [1, 2, 3, 4, 5, 6],
                  'dtc__max_depth': [3, 4, 5],
                  'dtc__criterion': ['gini', 'entropy']}
        grid = GridSearchCV(self.dtc_pipe, param_grid, cv=5, refit=True, verbose=2, n_jobs=-1)
        grid.fit(X, y)
        # Save results
        df_results = pd.DataFrame.from_dict(data=grid.cv_results_)
        df_results.to_csv("results\\dtc.csv",index=False)
        print(grid.best_params_)
        # Take best model
        self.dtc_pipe = grid.best_estimator_
        # Plot the dtc
        #plot_dtc(self.dtc_pipe['dtc'])
        # Save the model
        with open("models\\dtc_pipe.pkl", 'wb') as file:
            pickle.dump(self.dtc_pipe, file)

        # Fit SVC
        # Do GridSearch to get best model
        param_grid = {'svc__C': [10, 100, 1000, 10000],
                  'svc__gamma': [0.01, 0.001, 0.0001, 0.00001],
                  'svc__degree': [2, 3],
                  'svc__kernel': ['rbf', 'linear', 'poly']}
        grid = GridSearchCV(self.svc_pipe, param_grid, cv=5, refit=True, verbose=2, n_jobs=-1)
        grid.fit(X, y)
        # Save results
        df_results = pd.DataFrame.from_dict(data=grid.cv_results_)
        df_results.to_csv("results\\svc.csv",index=False)
        print(grid.best_params_)
        # Take best model
        self.svc_pipe = grid.best_estimator_
        # Save the model
        with open("models\\svc_pipe.pkl", 'wb') as file:
            pickle.dump(self.dtc_pipe, file)

        # Fit CNN
        # Do GridSearch to get best model
        param_grid = {'cnn__num_channels':[X.shape[2]], 
                  'cnn__len_input':[X.shape[1]], 
                  'cnn__num_classes':[np.unique(y).shape[0]],
                  'cnn__batch_size': [20, 30],
                  'cnn__num_filter': [4, 8, 16],
                  'cnn__num_layer': [1, 2],
                  'cnn__len_filter': [0.05, 0.1, 0.2]}  # len_filter is defined as fraction of input_len
        grid = GridSearchCV(self.cnn_pipe, param_grid, cv=5, refit=True, verbose=2, n_jobs=-1)
        grid.fit(X, y)
        # Save results
        df_results = pd.DataFrame.from_dict(data=grid.cv_results_)
        df_results.to_csv("results\\cnn.csv",index=False)
        print(grid.best_params_)
        # Take best model
        self.cnn_pipe = grid.best_estimator_
        # Save the model 
        self.cnn_pipe['cnn'].model.save("models\\cnn.h5")

        # Fit the Metaclassifiers 
        if stacked:
            # Get level 1 classifier predictions as training data
            X_stacked, y_stacked = kfoldcrossval(self, X, y, k=5)
            # Fit Meta DTC
            self.meta_dtc.fit(X_stacked, y_stacked)
            # Save the model
            with open("models\\meta_dtc.pkl", 'wb') as file:
                pickle.dump(self.meta_dtc, file)
            # Fit Meta SVC
            self.meta_svc.fit(X_stacked, y_stacked)
            # Save the model
            with open("models\\meta_svc.pkl", 'wb') as file:
                pickle.dump(self.meta_svc, file)

        return self

    def predict(self, X, voting='veto'):
        """
        Perform a classification on samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_channels)
            Test samples.
        voting: string
            Voting scheme to use
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Predictions
        y_pred_ens: array, shape (n_samples, 3)
            Predictions of the individual estimators
        """
        y_pred = np.empty(np.shape(X)[0])
        # Parallelize this part
        y_dtc = self.dtc_pipe.predict(X)
        y_svc = self.svc_pipe.predict(X)
        y_cnn = self.cnn_pipe.predict(X)

        y_pred_ens = np.stack([y_dtc, y_svc, y_cnn], axis=1).astype(int)

        if voting == 'veto':
            for i in range(np.shape(X)[0]):
                if y_dtc[i] == y_svc[i] == y_cnn[i]:
                    y_pred[i] = y_dtc[i]
                else:
                    y_pred[i] = -1

        if voting == 'democratic':
            for i in range(np.shape(X)[0]):
                y_pred[i] = np.argmax(np.bincount(y_pred_ens[i, :]))

        if voting == 'meta_dtc':
            y_pred = self.meta_dtc.predict(y_pred_ens)

        if voting == 'meta_svc':
            y_pred = self.meta_svc.predict(y_pred_ens)

        return y_pred, y_pred_ens


def kfoldcrossval(model, X, y, k=5):
    """
    Performs another cross-validation with the optimal models in order to
    get the level 1 predictions to train the meta classifier.
    Parameters
    ----------
    model: object
        Ensemble classifier object
    X : array-like of shape (n_samples, n_features, n_channels)
            Samples.
    y : array-like of shape (n_samples,)
            True labels for X.
    k: int
        Number of splits
    Returns
    -------
    X_stack: array-like of shape (n_samples, n_features)
        Level 1 predictions as training data for metaclassifier
    y_stack: array-like of shape (n_samples,)
        Targets for metaclassifier
    """
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X_stack = np.empty((0, 3))
    y_stack = np.empty((0,))

    # Make a copy of the already fitted classifiers (to not overwrite the originals)
    dtc_temp = clone(model.dtc_pipe)
    svc_temp = clone(model.svc_pipe)
    cnn_temp = clone(model.cnn_pipe)

    # Train classifiers agin in kfold crossvalidation to get level 1 predictions
    for train, test in kfold.split(X, y):
        # Train all models on train
        dtc_temp.fit(X[train], y[train])
        svc_temp.fit(X[train], y[train])
        cnn_temp.fit(X[train], y[train])
        # Test all on test
        y0 = dtc_temp.predict(X[test])
        y1 = svc_temp.predict(X[test])
        y2 = cnn_temp.predict(X[test])
        # Concatenate predictions of individual classifier
        a = np.stack((y0, y1, y2), axis=-1).astype(int)
        # Concatenate with predictions from other splits
        X_stack = np.vstack((X_stack, a))
        y_stack = np.hstack((y_stack, y[test]))
    return X_stack, y_stack


def build_cnn(num_filter, len_filter, num_layer, num_channels, len_input, num_classes):
    """
    Function returning a keras model.
    Parameters
    ----------
    num_filter: int
        Number of filters / kernels in the conv layer
    len_filter: float
        Length of the filters / kernels in the conv layer as fraction of inputlength
    num_layer: int
        Number of convlutional layers in the model
    num_channels: int
        Number of channels of the input
    len_input: int
        Number of dimensions of the input
    num_classes: int
        Number of classes in the dataset = Number of outputs
    Returns
    -------
    model: sequential keras model
        Keras CNN model ready to be trained
    """
    model = Sequential()
    # First Conv Layer
    model.add(Conv1D(filters=num_filter, kernel_size=int(len_filter*len_input), strides=1, padding="same",
                         activation='relu', input_shape=(len_input, num_channels), name='block1_conv1'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same", name='block1_pool'))
    # Other Conv Layers
    for l in range(2, num_layer + 1):
        model.add(Conv1D(filters=num_filter*l, kernel_size=int(len_filter * len_input), strides=1, padding="same",
                         activation='relu', name='block' + str(l) + '_conv1'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding="same", name='block' + str(l) + '_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(100, activation='relu', name='fc1'))
    model.add(Dense(num_classes, activation='softmax',name='predictions'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    plot_model(model,dpi = 300, show_shapes=True, to_file='models\\cnn.png')
    return model
