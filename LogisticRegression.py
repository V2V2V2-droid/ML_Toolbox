import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pca import *
import seaborn as sn



def add_intercept(x):
    return np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)

def sigmoid(x):
    return np.array([1/(1 + np.exp(i)) for i in x])


def standardized_matrix(x):
    x_mean = np.apply_along_axis(lambda u: round(np.mean(u), 3), 0, x)
    x_std = np.apply_along_axis(lambda u: round(np.std(u), 3), 0, x)
    return (x-x_mean)/x_std

# we need to standardize x before adding the 1s vector


def pred(beta, x):
    assert len(beta) == x.shape[1]
    # Calculate the matrix
    dot_result = np.matmul(beta, x.T)
    # Use sigmoid to get a result between 0 and 1
    return sigmoid(dot_result)

def neg_log_likelihood(y, y_pred):
    id = np.ones(len(y))
    """Our loss function: Negative Log Likelihood or log loss"""
    return - (np.matmul(y, np.log(y_pred).T) + np.matmul((id-y), np.log(id - y_pred).T))

def compute_grad(x,y_pred, y):
    # grad_j is the partial derivative of the cost function according to each beta_j
    # gradient is the vector of all partial derivatives
    # this is the formula of the derivative for the log loss function we defined
    # we take y_pred - y as we take the negative of the gradient: steepest descent
    grad = np.matmul(np.array(y_pred - y), x)
    return grad

# results are not great : around 70% 75% but we should improve the optimization
# algorithm cause it may be stuck in a local minima

class Logit:
    def __init__(self, x, y, beta=None, epochs=500, learning_rate=0.001):
        self.x = add_intercept(x)
        self.n, self.p = self.x.shape
        if beta is not None:
            assert beta.shape[1] == self.p, "weight vector must be of the same" \
                                                " lenght as nb of explanatory variables"
            self.beta = beta
        else:
            np.random.seed(12198)
            self.beta = np.random.randn(1, self.p)[0, :]
        # can imagine keeping the labels and encoding y
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.best_estimator = self.train()

    def train(self):
        for epoch in range(self.epochs):
            # calculate prediction
            y_pred = pred(self.beta, self.x)
            loss = neg_log_likelihood(self.y, y_pred)
            # calculate and print loss
            if (epoch +1) % 100 == True:
                print(f'Epoch {epoch} --> loss: {loss}')
            # compute gradient derivative vector of the loss function
            grad = compute_grad(x=self.x, y=self.y, y_pred=y_pred)
            # add a stopping mechanism before end of epochs if beta does not evolve anylonger
            if len(np.where(abs(grad*self.learning_rate) < 0.00001)[0]) > 0.9*self.p:
                print("the weights do not evolve anymore, stop process")
                break
            else:
                self.beta = [i + j*self.learning_rate for (i, j) in zip(self.beta, grad)]

        print(f'Best estimate for "beta": {self.beta}')
        return self.beta

    def predict(self, x_test):
        """returns the predicted value for an unknown x data point"""
        x_test =  add_intercept(standardized_matrix(x_test))
        y_ = pred(beta = self.best_estimator,x=x_test)
        # add the label as arg
        labels = ["good" if i >= 0.5 else "bad" for i in y_]
        print(labels)
        print(len(labels))
        return labels




