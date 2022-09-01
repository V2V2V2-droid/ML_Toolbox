import numpy as np
from numpy.linalg import norm


def update_groups(e, groups):
    # update the groups dict values by selection, from the matrix E: of vectors e_k
    # corresponding to the euclidian norm vector (n,1) of the normed distance matrixes
    # by taking for each row the min value of the row (min btw e_k vectors value for row i) and getting col number : k
    # k = closest center
    b = e.min(axis=1)
    for i in range(0, len(b)) :
        groups[i] = np.where(e[i, :] == b[i])[0][0]+1
    return groups


def euclidian_norm_on_row(x):
    return np.apply_along_axis(lambda u: round(norm(u), 2), 1, x)


def normed_matrix(x):
    x_mean = np.apply_along_axis(lambda u: round(np.mean(u), 3), 0, x)
    x_std = np.apply_along_axis(lambda u: round(np.std(u), 3), 0, x)
    return (x-x_mean)/x_std, x_mean, x_std



def Kdistance_vectors(centers, x, K):
    # X and centers should both be normed before running this function
    E =[]
    for k in range(1, K+1):
        d_k = x - centers[k]  # matrix of shape (n,m)
        e_k = euclidian_norm_on_row(d_k).reshape(-1, 1) # vector of shape (1,)
        E.append(e_k)
    return np.concatenate(E, axis=1)




def update_centers(x, groups, centers, K):
    # two dicts groups and centers
    # make the list of keys for each groups
    for k in range(1, K+1):
        index_k = [key for key, v in groups.items() if v == k]
        group_k = x[index_k]
        centers[k] = np.apply_along_axis(lambda u: np.mean(u), 0, group_k)
    return centers


def take_random_from_space(x):
    # if no data space is specified, we'll take the outer limit made by each point
    # no "furthest" point, but furthest possible points
    x_min = np.apply_along_axis(lambda u: np.min(u), 0, x)
    x_max = np.apply_along_axis(lambda u: np.max(u), 0, x)
    return np.random.uniform(x_min, x_max)


def take_random_from_normal(x):
    x_mu = np.apply_along_axis(lambda u: np.mean(u), 0, x)
    x_sigma = np.apply_along_axis(lambda u: np.std(u), 0, x)
    a = np.random.normal(x_mu, x_sigma, size=(1, x.shape[1]))
    return np.reshape(a, x.shape[1])


class Kmeans:
    def __init__(self, x, K, Niter=20, input_centers=None, mode_for_random="normal", normed=False):
        self.x = x
        self.K = K
        self.normed = normed
        self.mode_for_random = mode_for_random
        self.Niter = Niter
        self.n, self.m = x.shape
        self.input_centers = input_centers
        self.initial_centers = self.init_centers()
        self.x_normed, self.x_mean, self.x_std = normed_matrix(self.x)
        self.store_groups = {}
        self.store_centers = {}

    def init_centers(self):
        centers = {}
        if self.input_centers is None:
            for k in range(1, self.K + 1):
                if self.mode_for_random == "uniform":
                    centers[k] = take_random_from_space(self.x)
                else:
                    centers[k] = take_random_from_normal(self.x)
            return centers
        else:
            assert isinstance(self.input_centers, dict), "Initial centers param must be a dict of vectors of shape : (1,m)"
            assert len([*self.input_centers.keys()]) == self.K, "numbers of centers must be equal to K"
            return self.input_centers


    def run_kmeans(self):
        for j in range(self.Niter):
            if j == 0 :
                # if normed = True : compute E with normed x and normed initial centers
                if self.normed is True:
                  normed_centers = {k: (v - self.x_mean) / self.x_std for k, v in self.initial_centers.items()}
                  E = Kdistance_vectors(centers=normed_centers, x=self.x_normed, K=self.K)
                else:
                  E = Kdistance_vectors(centers=self.initial_centers, x=self.x, K=self.K)
                groups = update_groups(e=E, groups={})
                centers = update_centers(x=self.x, K=self.K, centers=self.initial_centers, groups=groups)
            else:
                if self.normed is True:
                  normed_centers = {k: (v - self.x_mean) / self.x_std for k, v in self.store_centers["iter n°{}".format(j)].items()}
                  E = Kdistance_vectors(centers=normed_centers, x=self.x_normed, K=self.K)
                else:
                  E = Kdistance_vectors(centers=self.store_centers["iter n°{}".format(j)], x=self.x, K=self.K)
                groups = update_groups(e=E, groups=self.store_groups["iter n°{}".format(j)])
                centers = update_centers(x=self.x, K=self.K,
                                              centers=self.store_centers["iter n°{}".format(j)],
                                              groups=groups)
            #print("centers at step {}".format(j))
            #print(self.centers)
            self.store_centers["iter n°{}".format(j+1)] = centers
            self.store_groups["iter n°{}".format(j+1)] = groups
        return groups, centers, self.store_groups, self.store_centers

