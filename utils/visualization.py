import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config

def create_confusion_matrix(y_true, y_pred, classes=None, output_path=None, figsize=(10, 8), normalize=False):
    """
    Create and save a confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        output_path: Path to save the visualization
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Path to the saved visualization
    """
    if classes is None:
        classes = config.DOCUMENT_CLASSES
    
    if output_path is None:
        output_path = os.path.join(config.DATA_DIR, "confusion_matrix.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure and plot
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Set up axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Annotate cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path)
    plt.close()
    
    return output_path 