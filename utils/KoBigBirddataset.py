import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KoBigBirdDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len,
                 pad, pair):
        
        self.sentences = [tokenizer(i[sent_idx], max_length=max_len, padding="max_length", truncation=True, return_tensors="pt") for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        data = {
            'input_data': self.sentences[i],
            'label_data': self.labels[i]
        }
        return data

    def __len__(self):
        return (len(self.labels))