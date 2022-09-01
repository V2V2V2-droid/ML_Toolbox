import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


def isSymmetricMatrix(u):
    if u.shape == np.transpose(u).shape:
        # print("matrix is square")
        if (np.transpose(u) == u).all():
            # print("the matrix is symmetric")
            return True
        else:
            print("the matrix is not symmetric")
            return False
    else:
        print("matrix is not square nor symmetric")
        return False


def normed_matrix(x):
    x_mean = np.apply_along_axis(lambda u: round(np.mean(u), 3), 0, x)
    x_std = np.apply_along_axis(lambda u: round(np.std(u), 3), 0, x)
    return (x-x_mean)/x_std


# the matrix that maximizes the variance is the projection of X*U with U the eigenvectors of the X_t*X_ matrix.
def covariance_matrix(x):
    """input must be the normed matrix"""
    return np.matmul(np.transpose(x), x)


def plot_explained_variance(variance_explained, cumulative):
    plt.bar(range(0, len(variance_explained)), variance_explained, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cumulative)), cumulative, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


class Pca:
    def __init__(self, x, d="90%"):
        self.x = x
        self.input_d = d
        self.normed_x = normed_matrix(self.x)
        self.s = covariance_matrix(x=self.normed_x)
        assert isSymmetricMatrix(self.s), "check input matrix, covariance matrix should be symmetric"
        self.w, self.u = eig(np.array(self.s, dtype=float))
        # in the case the rounding of eigenvalues brought up some imaginary numbers
        self.real_eigenvalues = [i.real for i in self.w]
        self.imag_eigenvalues = [i.imag for i in self.w]
        self.pca_proj = np.matmul(self.x, self.u)
        self.total_variance = np.sum(self.real_eigenvalues)
        self.variance_explained = [(i/self.total_variance) for i in sorted(self.real_eigenvalues, reverse=True)]
        self.d = self.define_d()
        self.reduced_x = self.pca_proj[:, :self.d]

    # find right number for d
    def define_d(self):
        cum_sum_exp = np.cumsum(self.variance_explained)
        if isinstance(self.input_d, int):
            return self.input_d
        elif self.input_d == "90%":
            plot_explained_variance(self.variance_explained, cum_sum_exp)
            # d with the 90% rule :
            d = len(cum_sum_exp[cum_sum_exp <= 0.9])
            print("90% rule : we choose d = {}".format(str(d)))
            plot_explained_variance(self.variance_explained[:d+1], cum_sum_exp[:d+1])
            return d

        elif self.input_d == "catell":
            print("catell returns d where the added variance explained between"
                  " component i and component i+1 is below 10% of max diff between components.")
            plot_explained_variance(self.variance_explained, cum_sum_exp)
            var_diff = cum_sum_exp[1:] - cum_sum_exp[:-1]
            threshold = 0.10 * max(var_diff)
            d = np.where(var_diff < threshold)[0][0]
            print("catell : we choose d = {}".format(str(d)))
            plot_explained_variance(self.variance_explained[:d+1], cum_sum_exp[:d+1])
            return d

        else:
            print("no reduction taking place, please give a d method = 90% for "
                  "the 90% rule or d = catell for catell algorithm.")
            return self.x.shape[1]



