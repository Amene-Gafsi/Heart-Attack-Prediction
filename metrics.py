import numpy as np
from helpers import predict
from processing import Threshold

"""
This file contains metrics and tool for model evaluation
"""
def accuracy(x, y, w):
    """Calculates the accuracy of the model.
    Args:
        x: data.
        y: labels.
        w: weights.
    Returns:
        accuracy: accuracy of the model.
    """
    y_pred = np.array(predict(w, x))
    y_true = np.array(y)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy


def f1(x, y, w, threshold=Threshold):
    """Calculates the F1 score of the model.
    Args:
        x: data.
        y: labels.
        w: weights.
    Returns:
        f1: F1 score of the model.
    """
    y_pred = predict(w, x, threshold)
    TP = np.sum((y == 1) & (y_pred == 1))
    FP = np.sum((y == -1) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == -1))
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )
    return f1


def find_best_threshold(x, y, w):
    """
    Prediction threshold tuning for best f1 score
    """
    thresholds = np.linspace(0, 1, 300)
    f1_score = 0
    for threshold in thresholds:
        f1_i = f1(x, y, w, threshold)
        if f1_i > f1_score:
            f1_score = f1_i
            best_threshold = threshold
    return best_threshold
