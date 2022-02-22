# -*- coding: utf-8 -*-
from __future__ import print_function

import random
import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N = len(x_train)
    L2 = -l2(x_train, np.transpose(test_datum.reshape(len(test_datum), 1)))
    num_A = []
    max_L2 = max(L2)
    for i in range(len(L2)):
        num_A.append(np.exp((L2[i] - max_L2)/(2*tau**2)))
    denom_A = sum(num_A)
    L = np.array(denom_A*num_A)
    A = np.diagflat(L)
    xAy = np.dot(np.dot(np.transpose(x_train), A), y_train)
    xAx = np.dot(np.dot(np.transpose(x_train), A), x_train)
    weights = np.linalg.solve(xAx + lam * np.identity(len(xAx)), xAy)

    return np.dot(np.transpose(test_datum), weights)


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    train_loss = []
    valid_loss = []
    indices = []
    N = len(x)
    for i in range(N):
        indices.append(i)
    random.shuffle(indices)
    training_indices = indices[:math.floor(val_frac*N)]
    validation_indices = indices[math.floor(val_frac%N):]
    val_data = []
    val_targets = []
    train_data = []
    train_targets = []
    for i in training_indices:
        val_data.append(x[i])
        val_targets.append(y[i])
    for i in validation_indices:
        train_data.append(x[i])
        train_targets.append(y[i])
    training_data = np.array(train_data)
    training_targets = np.array(train_targets)
    validation_data = np.array(val_data)
    validation_targets = np.array(val_targets)

    for t in taus:
        train_loss.append(training_loss(x, y, t))
        valid_loss.append(validation_loss(training_data, training_targets, t, validation_data, validation_targets))
    return np.array(train_loss), np.array(valid_loss)


def training_loss(x, y, tau):
    '''
    Input:
    :param x: N x d design matrix
    :param y: N x 1 targets vector
    :param tau: the tau value used in LRLS
    :return: the mean squared error for all training examples
    '''
    error = 0
    for i in range(len(y)):
        predicted = LRLS(x[i], x, y, tau)
        error += (1/len(y))*(y[i]-predicted)**2
    return error


def validation_loss(x, y, tau, validation_examples, validation_targets):
    '''
    Input:
    :param x: N x d design matrix
    :param y: N x 1 targets vector
    :param tau: the tau value used in LRLS
    :param validation_examples: the validation set
    :param validation_targets: the validation targets
    :return: the mean squared error for the validation set
    '''
    error = 0
    for i in range(len(validation_targets)):
        predicted = LRLS(validation_examples[i], x, y, tau)
        error += (1/len(validation_targets))*(validation_targets[i]-predicted)**2
    return error


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 1000)
    train_losses, val_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses, label = "training losses")
    plt.semilogx(val_losses, label = "validation losses")
    plt.legend
    plt.suptitle("Training and validation losses as a function of tau (logscale)")
    plt.show()
