import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def compute_metrics(logits, labels):
    # 로짓과 레이블을 기반으로 메트릭을 계산합니다.
    preds = torch.sigmoid(logits).round()  # 확률을 계산하고 0.5 기준으로 반올림하여 예측을 결정합니다.
    accuracy = accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
    # cm = confusion_matrix(labels.detach().cpu().numpy().argmax(axis=1), preds.detach().cpu().numpy().argmax(axis=1))
    return accuracy, precision, recall, f1

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