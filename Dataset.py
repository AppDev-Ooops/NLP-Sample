import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer='hfl/chinese-macbert-base'):
        self.labels = reviews['label']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.encodings = tokenizer(reviews['text'],
                                   padding='max_length', 
                                   truncation=True,
                                   max_length=128, return_tensors='pt')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': torch.tensor(self.labels[idx])}