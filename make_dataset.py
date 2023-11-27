import torch
import random
import os
import pandas as pd
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
    'EPOCHS':1,
    'LEARNING_RATE':1e-5,
    'PATIENCE':30,
    'BATCH_SIZE':16,
    'SEED':789
}

seed_everything(CFG['SEED']) # Seed 고정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv("dataset/df_all.csv", index_col=0) # 전체 데이터셋을 불러옴
dict_section = {val: idx for idx, val in enumerate(df['ipc코드'].str[:1].unique())} # section을 key로, 인덱스를 value로 하는 딕셔너리 생성
section_vector = [0] * 7 # section을 원-핫 인코딩한 벡터를 담을 리스트 생성
    
def preprocess(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.rename(columns={'re_초록': 'abstract', 'ipc코드': 'label'}) # column 이름 변경
    df_deduplicated = df.drop_duplicates(['출원번호', 'abstract', 'label']) # 중복 제거
    df_grouped = df_deduplicated.groupby(['출원번호', 'abstract'])['label'].apply(list).reset_index(name='new_label') # 출원번호, 초록 별로 묶어서 라벨을 리스트로 만듦
    df_grouped['section_only'] = df_grouped['new_label'].apply(lambda x: [label[:1] for label in x]) # 라벨에서 section만 추출
    df_grouped = df_grouped[~df_grouped['abstract'].str.contains('내용 없음')] # 내용 없음이라고 쓰여진 행 제거
    df_grouped = df_grouped.dropna() # 결측치 제거
    return df_grouped[['abstract', 'section_only']] # 필요한 컬럼만 추출

# labels encoding
def encode_labels(labels):
    for label in labels:
        section_vector[dict_section[label]] = 1 # 해당 section의 인덱스에 1을 넣음
    return section_vector

def make_dataset(train_path, valid_path, test_path):
    if not os.path.exists("dataset/train.pkl") and not os.path.exists("dataset/valid.pkl") and not os.path.exists("dataset/test.pkl"):
        # load raw data
        train = preprocess(train_path)
        valid = preprocess(valid_path)
        test = preprocess(test_path)

        # 인코딩된 라벨을 컬럼으로 추가
        train['section'] = train['section_only'].apply(encode_labels)
        valid['section'] = valid['section_only'].apply(encode_labels)
        test['section'] = test['section_only'].apply(encode_labels)

        train[['abstract', 'section']].to_pickle('dataset/train.pkl')
        valid[['abstract', 'section']].to_pickle('dataset/valid.pkl')
        test[['abstract', 'section']].to_pickle('dataset/test.pkl')
    else:
        train = pd.read_pickle("dataset/train.pkl")
        valid = pd.read_pickle("dataset/valid.pkl")
        test = pd.read_pickle("dataset/test.pkl")

    return train, valid, test
