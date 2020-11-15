import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def univariant(df, param, quantity='mean_test_score'):
    unique = df[param].unique()
    scores = []
    for i in unique:
        scores.append(df[df[param] == i][quantity].mean())

    plt.plot(unique, scores)
    plt.show()


def multivariant(df, param1, param2,quantity='mean_test_score'):
    unique1 = df[param1].unique()
    unique2 = df[param2].unique()
    unique1, unique2 = np.meshgrid(unique1, unique2)
    scores = np.zeros(unique1.shape)

    for i, p1 in enumerate(unique1[0]):
        for j, p2 in enumerate(unique2[0]):
            scores[i, j] = df[(df[param1] == p1) & (df[param2] == p2)][quantity].values.mean()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(unique1, unique2, scores, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel("Accuracy")
    plt.show()


df = pd.read_csv("..\\results\\cnn.csv")
univariant(df, param='param_cnn__len_filter',quantity='mean_score_time')