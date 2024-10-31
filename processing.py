"""
Some preprocessing tools
"""

import numpy as np

# Threshold is set to 0.7279 to maximize the F1 score.
Threshold = 0.7279


def preprocess_data(x_train, y_train, x_test):
    """Preprocesses the data by undersampling the majority class, replacing missing values, normalizing features, polynomial expansion, and adding bias column.
    Args:
        x_train: numpy array of features.
        y_train: numpy array of labels.
        x_test: numpy array of features.
    Returns:
        xtrain: numpy array of features with balanced classes, normalized, polynomial expanded, and with bias column.
        ytrain: numpy array of labels with balanced classes.
        xtest: numpy array of features with normalized, polynomial expanded, and with bias column.
        imbalanced_x: numpy array of features with normalized, polynomial expanded, and with bias column.
    """
    # Undersample the majority class to balance the dataset
    x_train_balanced, ytrain = undersample(x_train, y_train)

    # Replace missing values by the median of the training set
    x_train_clean, x_test_clean, imbalanced_x = impute_missing_with_train_medians(
        x_train_balanced, x_test, x_train
    )

    # Normalize features using z-score standardization
    xtrain, xtest, imbalanced_x = normalize_features(
        x_train_clean, x_test_clean, imbalanced_x
    )

    # Polynomial expansion of degree 2
    xtrain_poly, xtest_poly, imbalanced_x_poly = (
        polynomial_expansion(xtrain),
        polynomial_expansion(xtest),
        polynomial_expansion(imbalanced_x),
    )

    # Add bias column to the datasets
    xtrain = add_bias_column(xtrain_poly)
    xtest = add_bias_column(xtest_poly)
    imbalanced_x = add_bias_column(imbalanced_x_poly)

    return xtrain, ytrain, xtest, imbalanced_x


def impute_missing_with_train_medians(x_train, x_test, imbalanced_x):
    """Replaces missing values in training, testing, and original datasets with the median values of the training set.
    Args:
        x_train : numpy array with possible missing values represented as NaN.
        x_test : numpy array with possible missing values represented as NaN.
        imbalanced_x : numpy array with possible missing values represented as NaN.
    Returns:
        x_train: numpy array with missing values replaced by the median values of the training set.
        x_test: numpy array with missing values replaced by the median values of the training set.
        imbalanced_x: numpy array with missing values replaced by the median values of the training set.
    """

    medians = np.nanmedian(x_train, axis=0)

    nan_train = np.isnan(x_train)
    x_train[nan_train] = np.take(medians, np.where(nan_train)[1])

    nan_test = np.isnan(x_test)
    x_test[nan_test] = np.take(medians, np.where(nan_test)[1])

    nan_imbalanced_x = np.isnan(imbalanced_x)
    imbalanced_x[nan_imbalanced_x] = np.take(medians, np.where(nan_imbalanced_x)[1])

    return x_train, x_test, imbalanced_x


def undersample(x, y, majority_class=-1, minority_class=1):
    """Undersamples the majority class to balance the dataset.
    Args:
        x: numpy array of features.
        y: numpy array of labels.
        majority_class: label of the majority class.
        minority_class: label of the minority class.
    Returns:
        x: numpy array of features with balanced classes.
        y: numpy array of labels with balanced classes.
    """
    np.random.seed(12)

    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]

    num_to_keep = len(minority_indices)
    undersample_indices = np.random.choice(
        majority_indices, size=num_to_keep, replace=False
    )
    new_indices = np.concatenate([minority_indices, undersample_indices])

    np.random.shuffle(new_indices)

    return x[new_indices], y[new_indices]


def normalize_features(x_train, x_test, imbalanced_x):
    """Normalizes the training, testing, and original datasets using the z-score standardization.
    Args:
        x_train: numpy array of features.
        x_test: numpy array of features.
        imbalanced_x: numpy array of features.
    Returns:
        x_train_normalized: normalized numpy array of features.
        x_test_normalized: normalized numpy array of features.
        imbalanced_x_normalized: normalized numpy array of features.
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1

    x_train_normalized = (x_train - mean) / std
    x_test_normalized = (x_test - mean) / std
    imbalanced_x_normalized = (imbalanced_x - mean) / std

    return x_train_normalized, x_test_normalized, imbalanced_x_normalized


def polynomial_expansion(x, degree=2):
    """Adds columns to the dataset by raising the original features to the polynomial of the degree.
    Args:
        x: numpy array of features.
        degree: degree of the polynomial expansion.
    Returns:
        expanded_x: numpy array of features with additional columns
    """
    expanded_x = x
    for d in range(2, degree + 1):
        expanded_x = np.concatenate((expanded_x, x**d), axis=1)
    return expanded_x


def add_bias_column(x):
    """Adds a bias column to the dataset.
    Args:
        x: numpy array of features.
    Returns:
        x: numpy array of features with an additional bias column.
    """
    ones_column = np.ones((x.shape[0], 1))
    x = np.hstack([ones_column, x])
    return x
