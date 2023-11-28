import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

import numpy as np

def accuracy_multilabel(y_true, y_pred):
    """Compute the accuracy for multi-label classification."""
    correct_predictions = np.equal(y_true, y_pred).all(axis=1)
    accuracy = correct_predictions.mean()
    return accuracy

def precision_recall_f1_multilabel(y_true, y_pred):
    """Compute precision, recall, and F1-score for multi-label classification."""
    true_positives = np.logical_and(y_pred == 1, y_true == 1).sum(axis=0).astype(float)
    predicted_positives = y_pred.sum(axis=0).astype(float)
    actual_positives = y_true.sum(axis=0).astype(float)

    precision = np.divide(true_positives, predicted_positives, out=np.zeros_like(true_positives), where=predicted_positives != 0)
    recall = np.divide(true_positives, actual_positives, out=np.zeros_like(true_positives), where=actual_positives != 0)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # 1e-7 to avoid division by zero

    # Averaging across all labels
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_score_avg = np.mean(f1_score)

    return precision_avg, recall_avg, f1_score_avg

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint/best_results.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss