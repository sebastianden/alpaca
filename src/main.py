import numpy as np
import pandas as pd
from utils import split_df, TimeSeriesResampler, plot_confusion_matrix, Differentiator
from alpaca import Alpaca
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    """
    IMPORT YOUR DATA HERE
    X, y = 
    DEFINE RESAMPLING LENGTH IF NEEDED
    sz = 
    """
    
    # Turn y to numpy array
    y = np.array(y)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Pipeline example
    alpaca = Pipeline([('resampler', TimeSeriesResampler(sz=sz)),('alpaca', Alpaca())])
    alpaca.fit(X_train, y_train)
    
    """
    # Example with additional channel derived from channel 0
    alpaca = Pipeline([('resampler', TimeSeriesResampler(sz=sz)),
                       ('differentiator',Differentiator(channel=0)),
                       ('alpaca', Alpaca())])
    """

    y_pred_bin_veto, y_pred_veto = alpaca.predict(X_test, voting="veto")
    y_pred_bin_dem, y_pred_dem = alpaca.predict(X_test, voting="democratic")
    y_pred_bin_meta_dtc, y_pred_meta_dtc = alpaca.predict(X_test, voting="meta_dtc")
    y_pred_bin_meta_svc, y_pred_meta_svc = alpaca.predict(X_test, voting="meta_svc")

    # Store all results in a dataframe
    y_pred_indiv = np.column_stack((y_pred_bin_veto, y_pred_veto,y_pred_bin_dem, y_pred_dem, y_pred_bin_meta_dtc,
                                    y_pred_meta_dtc, y_pred_bin_meta_svc, y_pred_meta_svc, y_test)).astype(int)
    df_results = pd.DataFrame(y_pred_indiv, columns = ['y_pred_bin_veto', 'y_pred_veto','y_pred_bin_dem', 
                                                       'y_pred_dem', 'y_pred_bin_meta_dtc','y_pred_meta_dtc', 
                                                       'y_pred_bin_meta_svc', 'y_pred_meta_svc', 'y_true'])
    df_results.to_csv("results\\y_pred_total.csv",index=False)
    print("TEST FINISHED SUCCESSFULLY")

