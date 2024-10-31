import numpy as np
from helpers import *
from plots import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    loss = compute_loss(y, tx, initial_w)
    ws = [initial_w]
    losses = [loss]
    w = initial_w
    print("testing")
    for n_iter in range(max_iters):

        # compute gradient and loss
        grad = compute_gradient(y, tx, w)

        # update w
        w = w - gamma * grad

        # store w and loss
        loss = compute_loss(y, tx, w)
        ws.append(w)
        losses.append(loss)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    batch_size = 1

    # Define parameters to store w and loss
    loss = compute_loss(y, tx, initial_w)
    ws = [initial_w]
    losses = [loss]
    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):

            w = w - gamma * compute_stoch_gradient(y_batch, tx_batch, w)

            loss = compute_loss(y, tx, w)

            ws.append(w)
            losses.append(loss)

    return w, loss


def least_squares(y, tx):
    """Calculates the least squares solution.
       returns optimal weights, and MSE.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: mean squared error, scalar.
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """Calculates ridge regression.
       returns optimal weights, and MSE.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: mean squared error, scalar.
    """
    w = np.linalg.solve(
        tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T.dot(y)
    )
    mse = compute_loss(y, tx, w)
    return w, mse


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, plot=False):
    """
    Implementations of regularized logistic regression using gradient descent

    Input:
        - y         = the label
        - tx        = the samples
        - initial_w = the initial weights
        - max_iters = the maximum number of iterations
        - gamma     = learning rate (step size)
        - lambda_   = prefactor L2 regularization

    Output:
        - loss      = the final loss after traning
        - w         = the final training weights
    """

    loss = compute_reg_logistic_regression_loss(y, tx, initial_w, lambda_)
    w = initial_w
    losses = []
    for iter in range(max_iters):
        # Compute the gradient and the loss
        gradient = compute_reg_logistic_regression_gradient(y, tx, w, lambda_)

        # Update the weights
        w -= gamma * gradient
        # Return the final weights and the last loss ---> update loss
        loss = compute_reg_logistic_regression_loss(y, tx, w, lambda_)
        losses.append(loss)
    if plot:
        plot_gradient_loss_evolution(losses, max_iters, lambda_, gamma)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, plot=False):
    """
    Implementations of logistic regression using gradient descent

    Input:
        - y         = the label
        - tx        = the samples
        - initial_w = the initial weights
        - max_iters = the maximum number of iterations
        - gamma     = learning rate (step size)

    Output:
        - loss      = the final loss after traning
        - w         = the final training weights
    """
    w, loss = reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma, plot=plot)
    return w, loss
