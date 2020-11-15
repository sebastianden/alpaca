from alpaca import Alpaca
from utils import to_time_series_dataset, to_dataset, split_df, TimeSeriesResampler 
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

max_sample = 20

for dataset in ['uc2']:
    if dataset == 'uc1':
        X, y = split_df(pd.read_pickle('..\\data\\df_uc1.pkl'),
                        index_column='run_id',
                        feature_columns=['fldPosition', 'fldCurrent'],
                        target_name='target')
        y = np.array(y)
        # Length of timeseries for resampler and cnn
        sz = 38
        # Number of channels for cnn
        num_channels = len(X[0][0])
        # Number of classes for cnn
        num_classes = np.unique(y).shape[0]
    if dataset == 'uc2':
        X, y = split_df(pd.read_pickle('..\\data\\df_uc2.pkl'),
                        index_column='run_id',
                        feature_columns=['position', 'force'],
                        target_name='label')
        y = np.array(y)
        # Length of timeseries for resampler and cnn
        sz = 200
        # Number of channels for cnn
        num_channels = len(X[0][0])
        # Number of classes for cnn
        num_classes = np.unique(y).shape[0]

    resampler = TimeSeriesResampler(sz=sz)
    alpaca = Pipeline([('resampler', resampler),
                       ('classifier', Alpaca())])
    alpaca.fit(X, y, classifier__stacked=False, classifier__n_clusters=200)

    # Measure time for single sample processing
    t = []
    for i in range(1, max_sample+1):
        for j in range(10):
            rand = np.random.randint(2000)
            sample = np.transpose(to_time_series_dataset(X[rand]), (2, 0, 1))
            start = time.process_time()
            for k in range(100):
                for l in range(i):
                    y_pred_bin, y_pred = alpaca.predict(sample, voting='veto')
            end = time.process_time()
            t.append([i, (end-start)/100, 'single'])

    # Measure time for batch processing of multiple sample numbers
    for i in range(1, max_sample+1):
        for j in range(10):
            rand = np.random.randint(2000)
            if i == 1:
                sample = np.transpose(to_time_series_dataset(X[rand]), (2, 0, 1))
            else:
                sample = to_dataset(X[rand:rand+i])

            start = time.process_time()
            for k in range(100):
                y_pred_bin, y_pred = alpaca.predict(sample, voting='veto')
            end = time.process_time()
            t.append([i, (end-start)/100, 'batch'])

    df = pd.DataFrame(t, columns=['Sample Number', 'Time', 'Type'])
    df.to_csv("..\\results\\Time_"+dataset+".csv")




