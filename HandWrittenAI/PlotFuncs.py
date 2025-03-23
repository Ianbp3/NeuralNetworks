import matplotlib.pyplot as plt
import numpy as np

def loss_epochs(losses, n_epochs):
    epochs = list(range(1, n_epochs+1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, color='red', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def accu_epochs(accuracies, n_epochs):
    epochs = list(range(1, n_epochs + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accuracies, color='blue', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def correlation(accuracies, losses):
    accuracies = np.array(accuracies)
    losses = np.array(losses)
    correlation = np.corrcoef(accuracies, losses)[0, 1]
    print(f"Correlation: {correlation:.4f}")

def vis_correlation(accuracies, losses):
    plt.figure(figsize=(6, 4))
    plt.scatter(losses, accuracies, color='purple', marker='o')
    plt.title('Accuracy vs. Loss')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def loss_acc(accuracies, losses):
    fig, ax1 = plt.subplots()

    ax1.plot(accuracies, label='Accuracy', color='blue', marker='o')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(losses, label='Loss', color='red', marker='o')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.invert_yaxis()

    plt.title('Accuracy ↑ vs Loss ↓')
    plt.grid(True)
    plt.show()