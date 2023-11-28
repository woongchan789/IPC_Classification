from utils import EarlyStopping
from make_dataset import make_dataset
from model import CustomClassifier
from dataset import CustomDataset
from train import fit

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
import random
import os
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# wandb 초기화
wandb.init(project='ipc_section_classification')

# torch 버전 및 GPU 사용 가능 여부 확인
print('==================================')
print('torch version: ', torch.__version__)
print('Cuda available: ',torch.cuda.is_available())
print('==================================')

# 데이터셋 생성
print('Dataset Loading...')
train, valid, test = make_dataset('dataset/Train.csv', 'dataset/Valid.csv', 'dataset/Test.csv')
print('Dataset Loaded!')
print('==================================')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
    'EPOCHS':50,
    'LEARNING_RATE':1e-3,
    'PATIENCE':10,
    'BATCH_SIZE':64,
    'SEED':789
}

seed_everything(CFG['SEED']) # Seed 고정
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Early Stopping 인스턴스 초기화
early_stopping = EarlyStopping(patience=CFG['PATIENCE'], verbose=True, path='checkpoint/checkpoint.pt')

num_section_classes = 7

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')
model = CustomClassifier('klue/roberta-small', num_section_classes, device)

train_dataset = CustomDataset(train, tokenizer)
valid_dataset = CustomDataset(valid, tokenizer)
test_dataset = CustomDataset(test, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

optimizer = AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
loss_fn = nn.BCEWithLogitsLoss()  # 멀티 레이블 분류를 위한 손실 함수

fit(model, train_dataloader, valid_dataloader, loss_fn, CFG['EPOCHS'], optimizer, early_stopping, device)

# Weights & Biases 세션을 종료합니다.
wandb.finish()