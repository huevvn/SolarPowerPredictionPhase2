# evaluation.py - all metrics and plots

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# classification metrics
def clf_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'error_rate': 1 - acc,
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

# regression metrics
def reg_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # willmott index
    num = np.sum((y_pred - y_true) ** 2)
    denom = np.sum((np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2)
    willmott = 1 - (num / denom) if denom != 0 else 0
    
    # nash-sutcliffe efficiency
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # legates-mccabe index
    num_lm = np.sum(np.abs(y_pred - y_true))
    denom_lm = np.sum(np.abs(y_true - y_true.mean()))
    legates_mccabe = 1 - (num_lm / denom_lm) if denom_lm != 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'willmott': willmott,
        'nse': nse,
        'legates_mccabe': legates_mccabe
    }

def plot_confusion(y_true, y_pred, labels, path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_roc(y_true, y_proba, classes, path=None):
    # roc curve for multiclass
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        if y_proba.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores, title, path=None):
    # shows overfitting/underfitting
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores, 'o-', label='Training')
    plt.plot(train_sizes, test_scores, 'o-', label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve - {title}')
    plt.legend()
    plt.grid(True)
    
    # check fit status
    plt.text(0.5, 0.1, status, transform=plt.gca().transAxes, fontsize=12, color='red')
    
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    
    return status
