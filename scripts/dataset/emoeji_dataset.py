import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


"""
Dataset definition for MoE training
"""

class EmojiMoEDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128, desc_limit=2):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desc_limit = desc_limit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # concatenate first 3 descriptions
        desc = " ".join(item['generated_description'][:self.desc_limit])
        sent2 = item['sent2']
        label = int(item['label'])
        mapping = {"Metaphorical": 0, "Single": 1, "Semantic list": 2, "Reduplication": 3, "Direct": 4}
        strategy = mapping[int(item['strategy'])]

        # Tokenize sentence pair
        encoded = self.tokenizer(
            desc, sent2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'strategy': torch.tensor(strategy, dtype=torch.long)
        }

