import torch
import pandas as pd
import numpy as np
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from model import CustomClassifier
from utils import calculate_multilabel_metrics
from tqdm import tqdm

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

train = pd.read_pickle('dataset/train.pkl')
val = pd.read_pickle('dataset/valid.pkl')
test = pd.read_pickle('dataset/test.pkl')

data = pd.concat([train, val, test], ignore_index=True)

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')
dataset = CustomDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)  

model = CustomClassifier('klue/roberta-small', 7, device)
model.load_state_dict(torch.load('/home/woongchan/Workspace/지재권/ipc_section_classification/results2/model_state_dict.pth'))

total_loss = 0
total_section_acc = []
total_section_precison = []
total_section_recall = []
total_section_f1 = []
y_preds = []

model.eval()
with torch.no_grad():
    with tqdm(dataloader, total=len(dataloader)) as t:
        for batch in t:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            section_labels = batch['section'].to(device)

            outputs = model(input_ids, attention_mask)
            
            final_outputs = (outputs['section_output'].detach().cpu().numpy() > 0.5).astype(int)
            y_preds.append(final_outputs)
            
            for i in range(len(final_outputs)):
                if final_outputs[i].sum() == 0:
                    final_outputs[i][np.argmax(outputs['section_output'].detach().cpu().numpy()[i])] = 1
                    
            eval_dict = calculate_multilabel_metrics(section_labels.detach().cpu().numpy(), final_outputs)
            total_batch_acc,total_batch_precision,total_batch_recall,total_batch_f1 = eval_dict['accuracy'], eval_dict['precision'], eval_dict['recall'], eval_dict['f1_score']
            total_section_acc.append(total_batch_acc)
            total_section_precison.append(total_batch_precision)
            total_section_recall.append(total_batch_recall)
            total_section_f1.append(total_batch_f1)
            
            t.set_postfix(Acc=total_batch_acc, Prec=total_batch_precision, Rec=total_batch_recall, F1=total_batch_f1)

y_preds_flat = [item for sublist in y_preds for item in sublist]
data['section_pred'] = y_preds_flat
data['section'] = data['section'].apply(np.array)
data.to_pickle('inference_results/result_not_including_eng.pkl')

y_true = np.array(data['section'].to_list())
y_pred = np.array(data['section_pred'].to_list())

eval_dict = calculate_multilabel_metrics(y_true, y_pred)

print('=====================================================================================================')
print(f"Test")
print(f"Section - Accuracy: {eval_dict['accuracy']}, Precision: {eval_dict['precision']}, Recall: {eval_dict['recall']}, F1: {eval_dict['f1_score']}")
print('=====================================================================================================')

# Mapping from index to label
index_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

# Function to convert binary list to label list
def binary_to_labels(binary_list):
    labels = [index_to_label[idx] for idx, value in enumerate(binary_list) if value == 1]
    return labels

# Apply the function to the 'section' and 'section_pred' columns
data['section_labels'] = data['section'].apply(binary_to_labels)
data['section_pred_labels'] = data['section_pred'].apply(binary_to_labels)

data.to_pickle('inference_results/result_including_eng.pkl')
