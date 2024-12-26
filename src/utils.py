import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_model(model, filepath):
    """Saves the model to the specified filepath."""
    model.save(filepath)
    print(f"Model saved at: {filepath}")

def load_model(filepath):
    """Loads a model from the specified filepath."""
    model = tf.keras.models.load_model(filepath)
    print(f"Model loaded from: {filepath}")
    return model

def plot_confusion_matrix(y_true, y_pred, classes, results_dir):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.show()
