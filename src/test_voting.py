from alpaca import Alpaca
from utils import load_test, split_df, TimeSeriesResampler,confusion_matrix
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


if __name__ == '__main__':

    X, y = load_test()
    # Length of timeseries for resampler and cnn
    sz = 230
    # Number of channels for cnn
    num_channels = X.shape[-1]
    # Number of classes for cnn
    num_classes = np.unique(y).shape[0]
    classes = np.array(["0", "1", "2", "3", "4", "?"])

    repetitions = 1

    results = []
    outliers = np.empty((0, 230*3+5))

    for r in range(repetitions):
        print("Repetition #",r)

        X, y = shuffle(X, y, random_state=r)
        # Turn y to numpy array
        y = np.array(y)
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=r)

        for votingstr in ["democratic", "veto", "stacked_svc", "stacked_dtc"]:

            if votingstr == 'stacked_svc':
                meta = 'svc'
            elif votingstr == 'stacked_dtc':
                meta = 'dtc'

            if votingstr == 'stacked_svc' or votingstr == 'stacked_dtc':
                voting = 'stacked'
                stacked = True
            else:
                stacked = False
                voting = votingstr
                meta = None

            # Build pipeline from resampler and estimator
            resampler = TimeSeriesResampler(sz=sz)
            alpaca = Pipeline([('resampler', resampler),
                              ('classifier', Alpaca())])
            alpaca.fit(X_train, y_train, classifier__stacked=stacked, classifier__n_clusters=100)
            y_pred_bin, y_pred = alpaca.predict(X_test, voting=voting)

            # Plot confusion matrix (Binary)
            y_test_bin = np.copy(y_test)
            y_test_bin[y_test_bin > 0] = 1

            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()

            # Append overall error
            results.append([votingstr, r, 'err', (fp+fn)/(tn+fp+fn+tp)])

            # Append false negative rate
            results.append([votingstr, r, 'fnr', fn/(fn+tp)])

            # Append false positive rate
            results.append([votingstr, r, 'fpr', fp/(fp+tn)])

            # Save misclassified samples (with y_pred,y_pred_bin, y_true, and voting scheme)
            idx = np.where(y_test_bin != y_pred_bin)
            # Flattened curves
            curves = X_test[idx].transpose(0, 2, 1).reshape(X_test[idx].shape[0],-1)
            vote_type = np.array([votingstr for i in range(idx[0].shape[0])]).reshape((-1,1))
            wrong = np.hstack([curves, y_pred[idx].reshape((-1,1)),y_test[idx].reshape((-1,1)),
                               y_pred_bin[idx].reshape((-1,1)),y_test_bin[idx].reshape((-1,1)), vote_type])
            outliers = np.vstack((outliers,wrong))


    df = pd.DataFrame(outliers)
    df.to_csv("..\\results\\OutliersVotingTest.csv")

    df = pd.DataFrame(results, columns=['voting', 'test', 'metric', 'value'])
    df.to_csv("..\\results\\VotingTest.csv")


