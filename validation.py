"""
Models validation methods 
"""

from implementations import *


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold.
    Args:
        y: labels.
        k_fold: number of folds.
        seed: random seed.
    Returns:
        k_indices: indices for k-fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, gamma, max_iters):
    """Performs cross-validation.
    Args:
        y: labels.
        x: data.
        k_indices: indices for k-fold.
        k: index of the fold.
        lambda_: regularization parameter.
        gamma: learning rate.
        max_iters: maximum number of iterations.
    Returns:
        loss_train: loss of the training data.
        loss_test: loss of the testing data.
        w: weights.
    """
    test_indices = k_indices[k]
    train_indices = np.hstack([k_indices[i] for i in range(len(k_indices)) if i != k])
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    w, loss_train = reg_logistic_regression(
        y_train, x_train, lambda_, np.zeros(x_train.shape[1]), max_iters, gamma
    )
    loss_test = compute_reg_logistic_regression_loss(y_test, x_test, w, lambda_)
    return loss_train, loss_test, w


def cross_validation_demo(k_fold, lambdas, gammas, max_iters, x_train, y_train_01):
    """Performs cross-validation to find the best hyperparameters and plots the loss progression across rounds."""
    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y_train_01, k_fold, seed)
    # define lists to store the loss of training data and test data for each combination
    best_loss = np.inf
    best_lambda = None
    best_gamma = None
    best_w = None

    for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            loss_trs = np.empty(k_fold)
            loss_tes = np.empty(k_fold)
            # For each fold, perform cross-validation and store the losses
            for k in range(k_fold):
                loss_tr, loss_te, w = cross_validation(
                    y_train_01, x_train, k_indices, k, lambda_, gamma, max_iters
                )
                loss_trs[k] = loss_tr
                loss_tes[k] = loss_te

            # Calculate the mean test loss for this (lambda, gamma) pair
            mean_loss = np.mean(loss_tes)
            print(f"Lambda: {lambda_}, Gamma: {gamma}, Mean Test Loss: {mean_loss}")

            # Update the best parameters if the current mean loss is lower
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_lambda = lambda_
                best_gamma = gamma
                best_w = w

    return best_loss, best_lambda, best_gamma, best_w
