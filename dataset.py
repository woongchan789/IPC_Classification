import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['abstract']
        section = row['section']

        # Text를 토크나이징합니다.
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        # 데이터를 텐서로 변환합니다.
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        section = torch.tensor(section, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'section': section
        }