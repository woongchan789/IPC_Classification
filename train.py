from utils import accuracy_multilabel, precision_recall_f1_multilabel, EarlyStopping
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
import json
import os
import warnings
warnings.filterwarnings('ignore')

def fit(model, train_dataloader, valid_dataloader, loss_fn, epochs, optimizer, early_stopping, device):
    
    best_acc = 0
    
    # 훈련 루프
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_section_acc = []
        train_section_precison = []
        train_section_recall = []
        train_section_f1 = []
        
        with tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {epoch+1}") as t:
            for batch in t:
                # 배치를 GPU로 이동 (사용 가능한 경우)
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                section_labels = batch['section'].to(device)
                
                optimizer.zero_grad()
                
                # 모델을 통해 예측을 수행합니다.
                outputs = model(input_ids, attention_mask)
                
                # 손실을 계산합니다.
                loss = loss_fn(outputs['section_output'], section_labels)
                total_loss = loss
                
                # 역전파를 수행합니다.
                total_loss.backward()
                optimizer.step()
                
                train_total_loss += total_loss.item()
                train_batch_acc = accuracy_multilabel(section_labels.detach().cpu().numpy(), outputs['section_output'].detach().cpu().numpy())
                train_batch_precision, train_batch_recall, train_batch_f1 = precision_recall_f1_multilabel(section_labels.detach().cpu().numpy(), outputs['section_output'].detach().cpu().numpy())
                train_section_acc.append(train_batch_acc)
                train_section_precison.append(train_batch_precision)
                train_section_recall.append(train_batch_recall)
                train_section_f1.append(train_batch_f1)
                
                t.set_postfix(Acc=train_batch_acc, Prec=train_batch_precision, Rec=train_batch_recall, F1=train_batch_f1)
            
        avg_train_total_loss = train_total_loss / len(train_dataloader)
        avg_train_section_metrics = np.mean([train_section_acc, train_section_precison, train_section_recall, train_section_f1], axis=1)
        
        print('=====================================================================================================')
        print(f"Training - Epoch {epoch+1}")
        print(f"Total Loss: {avg_train_total_loss}")
        print(f"Section - Accuracy: {avg_train_section_metrics[0]}, Precision: {avg_train_section_metrics[1]}, Recall: {avg_train_section_metrics[2]}, F1: {avg_train_section_metrics[3]}")
        print('=====================================================================================================')
        wandb.log({'Train Loss': avg_train_total_loss,
                   'Train Section Accuracy': avg_train_section_metrics[0],
                   'Train Section Precision': avg_train_section_metrics[1],
                   'Train Section Recall': avg_train_section_metrics[2],
                   'Train Section F1': avg_train_section_metrics[3]}, step=epoch)
        # 검증 루프
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            val_section_acc = []
            val_section_precison = []
            val_section_recall = []
            val_section_f1 = []

            with tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Validation Epoch {epoch+1}") as t:
                for batch in t:
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    section_labels = batch['section'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs['section_output'], section_labels)
                    
                    # 각 섹션별 메트릭을 계산합니다.
                    val_total_loss += loss.item()    
                    val_batch_acc = accuracy_multilabel(outputs['section_output'].detach().cpu().numpy(), section_labels.detach().cpu().numpy())
                    val_batch_precision, val_batch_recall, val_batch_f1 = precision_recall_f1_multilabel(outputs['section_output'].detach().cpu().numpy(), section_labels.detach().cpu().numpy())
                    val_section_acc.append(val_batch_acc)
                    val_section_precison.append(val_batch_precision)
                    val_section_recall.append(val_batch_recall)
                    val_section_f1.append(val_batch_f1)
                    
                    t.set_postfix(Acc=val_batch_acc, Prec=val_batch_precision, Rec=val_batch_recall, F1=val_batch_f1)
            
            avg_val_total_loss = val_total_loss / len(valid_dataloader)
            avg_val_section_metrics = np.mean([val_section_acc, val_section_precison, val_section_recall, val_section_f1], axis=1)

            print('=====================================================================================================')
            print(f"Validation - Epoch {epoch+1}")
            print(f"Total Loss: {avg_val_total_loss}")
            print(f"Section - Accuracy: {avg_val_section_metrics[0]}, Precision: {avg_val_section_metrics[1]}, Recall: {avg_val_section_metrics[2]}, F1: {avg_val_section_metrics[3]}")
            print('=====================================================================================================')
            wandb.log({'Valid Loss': avg_val_total_loss,
                    'Valid Section Accuracy': avg_val_section_metrics[0],
                    'Valid Section Precision': avg_val_section_metrics[1],
                    'Valid Section Recall': avg_val_section_metrics[2],
                    'Valid Section F1': avg_val_section_metrics[3]}, step=epoch)
            
            # 체크포인트 저장 및 Early Stopping 체크
            if best_acc < avg_val_section_metrics[0]:
                    # save results
                torch.save(model.state_dict(), os.path.join('results', 'model_state_dict.pth'))
                torch.save(optimizer.state_dict(), os.path.join('results', 'optimizer_state_dict.pth'))

                state = {'best_epoch': epoch,
                        'loss': avg_val_total_loss,
                        'best_acc': avg_val_section_metrics[0],
                        'best_precision': avg_val_section_metrics[1],
                        'best_recall': avg_val_section_metrics[2],
                        'best_f1': avg_val_section_metrics[3]
                        }
                
                json.dump(state, open(os.path.join('results', f'best_results.json'),'w'), indent=4)

            early_stopping(avg_val_total_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

# Weights & Biases 세션을 종료합니다.
wandb.finish()