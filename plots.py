import matplotlib.pyplot as plt
import numpy as np

"""
Find here the implementation of several plots
"""

def plot_gradient_loss_evolution(losses, max_iter, lambda_, gamma):
    """
    Plot the loss evolution during training
    """
    iterations = range(max_iter)
    # Plotting
    plt.figure(figsize=(20, 12))
    plt.plot(iterations, losses, label="Training Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss Progression for λ={lambda_}, γ={gamma}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_class_imbalance(y_train):
    """
    Plot the data imblance in the training data
    """
    # Plot class imbalance
    np.random.seed(12)
    # Count occurrences of each class
    unique, counts = np.unique(y_train, return_counts=True)
    print(unique)
    print(counts)

    # Plotting class imbalance
    plt.figure(figsize=(8, 5))
    plt.bar(
        unique, counts, color=["skyblue", "salmon"], tick_label=["Class -1", "Class 1"]
    )
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Imbalance in Training Data")
    plt.show()
