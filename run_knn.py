from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    k_values = [1, 3, 5, 7, 9]
    classification_rates = []

    for k in k_values:
        predicted_targets = knn(k, train_inputs, train_targets, valid_inputs)
        classified_count = 0
        for i in range(len(predicted_targets)):
            if predicted_targets[i] == valid_targets[i]:
                classified_count += 1
        classification_rate = classified_count/len(predicted_targets)
        classification_rates.append(classification_rate)

    plt.plot(k_values, classification_rates, 'ro')
    plt.xlabel("K Hyperparameter")
    plt.ylabel("Classification Rate")
    plt.suptitle("K-NN Classification Rate")
    plt.savefig("Q3knn.png")
    plt.show()


    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

# Q3 b)
# From the plot of classification rate on the validation set vs the k hyper parameter I can see
# that for k = 3, 5, 7, I got the same classification rate, 86%. For k = 1,
# it was 82% and for k = 9, it was 84% percent. The two extreme choices of k, 1 and 9 correspond to
# overfitting in the case of k = 1, and underfitting in the case of k = 9
# When k = 1 the classifier has worse performance when generalizing to the validation set because it is too sensitive to the training data
# and when k = 9 the classifier has a boundary that is oversimplified. Of the three choices of
# k that have the same classification rate, I would choose k = 7 since I believe it would be better
# at generalizing to new data since it is large enough to generalized away quirks that are specific to
# the training set.

if __name__ == "__main__":
    run_knn()
