from alpaca import Alpaca
from utils import to_time_series_dataset, split_df, TimeSeriesResampler, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import time
import numpy as np
import pandas as pd

# Variables
repetitions = 2

if __name__ == "__main__":

    # For both datasets
    for dataset in ['uc1']:
        print("Dataset: ", dataset)

        results = []
        #timing = []
        #outliers = []

        if dataset == 'uc1':
            X, y = split_df(pd.read_pickle('..\\data\\df_uc1.pkl'),
                            index_column='run_id',
                            feature_columns=['fldPosition', 'fldCurrent'],
                            target_name='target')
            # Length of timeseries for resampler and cnn
            sz = [38,41]
            # Number of channels for cnn
            num_channels = len(X[0][0])
            # Number of classes for cnn
            num_classes = np.unique(y).shape[0]

        elif dataset == 'uc2':
            X, y = split_df(pd.read_pickle('..\\data\\df_uc2.pkl'),
                            index_column='run_id',
                            feature_columns=['position', 'force'],
                            target_name='label')
            # Length of timeseries for resampler and cnn
            sz = [200]
            # Number of channels for cnn
            num_channels = len(X[0][0])
            # Number of classes for cnn
            num_classes = np.unique(y).shape[0]

        # For each repetition
        for r in range(repetitions):
            print("Repetition #", r)
            # For each resampling length
            for s in sz:
                print("Resampling size:", s)
                t_start = time.time()
                # Shuffle for Keras
                X, y = shuffle(X, y, random_state=r)
                # Turn y to numpy array
                y = np.array(y)
                # Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=r)

                alpaca = Pipeline([('resampler', TimeSeriesResampler(sz=s)),
                                   ('classifier', Alpaca())])
                alpaca.fit(X_train, y_train, classifier__stacked=False, classifier__n_clusters=200)

                # Prediction
                y_pred_bin, y_pred = alpaca.predict(X_test, voting="veto")
                y_test_bin = np.copy(y_test)
                y_test_bin[y_test_bin > 0] = 1

                # BINARY RESULTS (AD + ENSEMBLE)
                tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
                # Append overall error
                results.append([s, r, 'err_bin', (fp + fn) / (tn + fp + fn + tp)])
                # Append false negative rate
                results.append([s, r, 'fnr_bin', fn / (fn + tp)])
                # Append false positive rate
                results.append([s, r, 'fpr_bin', fp / (fp + tn)])

                # CLASSIFIER RESULTS
                y_pred_clf = np.copy(y_pred)
                y_pred_clf[y_pred_clf != 0] = 1     # Also turn classifier predictions to binary for cfm
                tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_clf).ravel()
                # Append overall error
                results.append([s, r, 'err_ens', (fp + fn) / (tn + fp + fn + tp)])
                # Append false negative rate
                results.append([s, r, 'fnr_ens', fn / (fn + tp)])
                # Append false positive rate
                results.append([s, r, 'fpr_ens', fp / (fp + tn)])
                """
                # TIMING
                sample = np.transpose(to_time_series_dataset(X_test[0]), (2, 0, 1))
                start = time.time()
                for i in range(100):
                    alpaca.predict(sample, voting='veto')
                end = time.time()
                timing.append([(end - start) * 10, s]) # in ms


                # SAVE OUTLIERS (with y_pred,y_pred_bin, y_true)
                idx = np.where(y_test_bin != y_pred_bin)
                # Flattened curves
                for i in idx[0]:
                    outliers.append([X_test[i],
                                     y_pred[i],
                                     y_test[i],
                                     y_pred_bin[i],
                                     y_test_bin[i]])
                """
                t_end = time.time()
                print("Substest finished, duration ",(t_end-t_start))

        # SAVE ALL RESULTS PER DATASET
        df = pd.DataFrame(results, columns=['resampling', 'test', 'metric', 'value'])
        df.to_csv("..\\results\\Test"+dataset+".csv")
        #df = pd.DataFrame(timing, columns=['time', 'resampling'])
        #df.to_csv("..\\results\\Timing"+dataset+".csv")
        #df = pd.DataFrame(outliers, columns=['sample', 'y_pred', 'y_test', 'y_pred_bin', 'y_test_bin'])
        #df.to_pickle("..\\results\\Outliers"+dataset+".pkl")


    #plot_confusion_matrix(y_test_bin.astype(int), y_pred_bin.astype(int), np.array(["0", "1"]), cmap=plt.cm.Blues)
    #plt.show()
    #plot_confusion_matrix(y_test.astype(int), y_pred.astype(int), np.array(["0", "1", "2", "3", "?"]), cmap=plt.cm.Greens)
    #plt.show()



