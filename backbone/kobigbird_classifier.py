import torch
from torch import nn

class KoBigBirdClassifier(nn.Module):
    def __init__(self,
                 kobigbird,
                 max_len,
                 batch_size,
                 device,
                 hidden_size = 768,
                 num_classes=19,
                 dr_rate=None,
                 freeze_opt=True
                 ):
        super(KoBigBirdClassifier, self).__init__()
        self.kobigbird = kobigbird
        self.dr_rate = dr_rate
        self.batch_size = batch_size
        self.device = device
                 
        self.classifier = nn.Linear(hidden_size*max_len, num_classes)
        
        if freeze_opt==True:
            for param in self.kobigbird.parameters():
                param.requires_grad = False
            
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_data):
        input_ids = input_data['input_ids']
        attention_mask = input_data['attention_mask']
        new_input_ids = input_ids.squeeze().to(self.device)
        new_attention_mask = attention_mask.squeeze().to(self.device)

        output = self.kobigbird(input_ids = new_input_ids, attention_mask = new_attention_mask)
        reshaped_tensor = output.last_hidden_state.view(self.batch_size, -1)

        if self.dr_rate:
            out = self.dropout(reshaped_tensor)

        return self.classifier(out)