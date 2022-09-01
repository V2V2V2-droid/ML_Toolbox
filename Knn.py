import  numpy as np
from pca import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from kmeans import euclidian_norm_on_row
from numpy.linalg import norm
import seaborn as sn


def standardized_matrix(x):
    x_mean = np.apply_along_axis(lambda u: round(np.mean(u), 3), 0, x)
    x_std = np.apply_along_axis(lambda u: round(np.std(u), 3), 0, x)
    return (x-x_mean)/x_std




# test array

a = np.array([[1, 0, 2],
            [2, 3, 1],
            [8, 5, 6],
            [0, 7, 3]])
b = np.array(["g", "g", "b", "b"])
t = np.array([3, 9, 5])
dist = a - t
e = euclidian_norm_on_row(dist)
ks = np.argsort(e)[:3]
m =b[ks]


class Knn:
    def __init__(self, k, x, y):
        self.y = y # y is y_train here
        self.x = x # x must be normed before
        self.k = k

    def compute_distances(self, datapoint):
        # must be np array types
        d = self.x - datapoint
        e = euclidian_norm_on_row(d)
        return e

    def closest_neighboors(self, datapoint):
        e_k = self.compute_distances(datapoint=datapoint)
        ks = np.argsort(e_k)[:self.k]

        return ks

    def assign_group(self, datapoint):
        ks = self.closest_neighboors(datapoint=datapoint)
        m = self.y[ks]
        group = max(m, key=list(m).count)
        return group

    def predict(self, datapoints):
        y_pred = []
        for i in range(len(datapoints)):
            y_pred.append(self.assign_group(datapoint=datapoints[i, :]))

        return np.array(y_pred)


    def score(self, y_pred, y_test):
        # 1d arrays for both
        return sum(np.array(y_pred == y_test).astype(int))/len(y_test)


