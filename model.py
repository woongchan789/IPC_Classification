import torch
from transformers import AutoModel
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

class CustomClassifier(nn.Module):
    def __init__(self, bert_model_name, num_class, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name).to(device)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.gradient_checkpointing_enable()  
        self.drop = nn.Dropout(p=0.1)
        self.device = device
        self.section_classifier = nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size).to(self.device),
                    nn.LayerNorm(self.bert.config.hidden_size).to(self.device),
                    nn.Dropout(p = 0.1),
                    nn.ReLU().to(self.device),
                    nn.Linear(self.bert.config.hidden_size, num_class).to(self.device))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8).to(self.device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(self.device)
        outputs = transformer_encoder(outputs.last_hidden_state)
        outputs = outputs[:,0]
        output = self.drop(outputs)
        
        # Section 분류를 위한 로직
        output_section = self.section_classifier(output)

        return {'section_output' : output_section}
