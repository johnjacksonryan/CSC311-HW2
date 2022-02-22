import numpy

from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    small_train_inputs, small_train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.007,
        "num_iterations": 15
    }
    weights = hyperparameters["weight_regularization"]*np.ones(M+1)
    weights_small = hyperparameters["weight_regularization"] * np.ones(M + 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    error_list_train = []
    error_list_valid = []
    error_list_train_small = []
    error_list_valid_small = []
    iteration_count = []
    # Loop for gradient descent and to record the training error and validation error on each iteration
    for t in range(hyperparameters["num_iterations"]):
        iteration_count.append(t)
        # Using the large training set
        info_train = logistic(weights, train_inputs, train_targets, hyperparameters)
        f_train = info_train[0]
        error_list_train.append(f_train)
        df = info_train[1]
        # Using the small training set
        info_train_small = logistic(weights_small, small_train_inputs, small_train_targets, hyperparameters)
        f_train_small = info_train_small[0]
        error_list_train_small.append(f_train_small)
        df_small = info_train_small[1]
        # Recording error on validation sets
        error_list_valid.append(logistic(weights, valid_inputs, valid_targets, hyperparameters)[0])
        error_list_valid_small.append(logistic(weights_small, valid_inputs, valid_targets, hyperparameters)[0])
        # Gradient descent to update the weights
        update = []
        update_small = []
        learn_rate = hyperparameters["learning_rate"]
        for i in range(len(weights)):
            update.append(weights[i] - learn_rate*df[i])
            update_small.append(weights_small[i] - learn_rate*df_small[i])
        weights = np.array(update)
        weights_small = np.array(update_small)

    # Plots
    # Large data set
   # plt.plot(iteration_count, error_list_train, label = "Training error")
   # plt.plot(iteration_count, error_list_valid, label = "Validation error")
   # plt.xlabel("Number of Iterations")
   # plt.ylabel("Loss")
   # plt.suptitle("Cross entropy from large training set as a function of iterations")
   # plt.legend()
   # plt.show()

    # Small data set
    plt.plot(iteration_count, error_list_train_small, label = "Training error")
    plt.plot(iteration_count, error_list_valid_small, label = "Validation error")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.suptitle("Cross entropy from small training set as a function of iterations")
    plt.legend()
    plt.show()

    # Error reporting
   # y = info_train[2]
   # eval = evaluate(train_targets, y)
   # print("Large data set training error")
   # print("Cross Entropy: ", eval[0])
   # print("Fraction correct: ", eval[1])
   # y = logistic(weights, valid_inputs, valid_targets, hyperparameters)[2]
   # eval = evaluate(valid_targets, y)
   # print("Large data set validation error")
   # print("Cross Entropy: ", eval[0])
   # print("Fraction correct: ", eval[1])
    y = info_train_small[2]
    eval = evaluate(small_train_targets, y)
    print("Small data set training error")
    print("Cross Entropy: ", eval[0])
    print("Fraction correct: ", eval[1])
    y = logistic(weights_small, valid_inputs, valid_targets, hyperparameters)[2]
    eval = evaluate(valid_targets, y)
    print("Small data set validation error")
    print("Cross Entropy: ", eval[0])
    print("Fraction correct: ", eval[1])

    # Test error with final hyperparameters
    test_inputs, test_targets = load_test()
   # y = logistic(weights, test_inputs, test_targets, hyperparameters)[2]
   # eval = evaluate(test_targets, y)
   # print("Large data set test error")
   # print("Cross Entropy: ", eval[0])
   # print("Fraction correct: ", eval[1])
    y = logistic(weights_small, test_inputs, test_targets, hyperparameters)[2]
    eval = evaluate(test_targets, y)
    print("Small data set test error")
    print("Cross Entropy: ", eval[0])
    print("Fraction correct: ", eval[1])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
