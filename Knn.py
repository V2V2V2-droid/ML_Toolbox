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



#y = np.array(y == "good").astype(int)

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

if __name__ == '__main__':
    wine = pd.read_csv("wine.csv")
    x = np.array(wine)[:, :-1]
    y = np.array(wine)[:, -1]

    #we obtain better results without PCA

    # let's try variable selection from the pairplot
    # remove residual sugar, chloride and fixed acidity
    x = x[:, [0,1, 2, 5, 6, 9, 10]]


    # PCA
    #pca = Pca(x, d=6)
    #r_proj = pca.reduced_x
    #f = lambda u: u.real 
    #x = f(r_proj)

    # standardize matrix x
    x = standardized_matrix(x)

    # train test split:
    x_train = x[:1550, :]
    y_train = y[:1550]
    x_test = x[1550:, :]
    y_test = y[1550:]


    # choose right k: even if that would need CV
    score = []
    for k in range(1, 15):
        print("k={}".format(k))
        A = Knn(x=x_train, k=k, y=y_train)
        y_pred = A.predict(datapoints=x_test)
        score.append(A.score(y_pred=y_pred, y_test=y_test))
    print(score)

    best_k = np.argmax(score)
    print(best_k)

# arrives at about 75 score max: even if we should refine k using cross validation


    # plot true data

    #df = pd.DataFrame(r_proj)
    #df["quality"] = wine["quality"]
    #sn.pairplot(df, hue="quality")
    #plt.show()

    # plot the y_test true

    #df1 = pd.DataFrame(x_test)
    #df1["quality"] = y_test
    #sn.pairplot(df1, hue="quality")
    #plt.show()

    # plot out y_pred

    #df2 = pd.DataFrame(x_test)
    #df2["quality"] = y_pred
    #sn.pairplot(df2, hue="quality")
    #plt.show()
