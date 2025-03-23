# =============================
# 1. Data Preparation
# =============================
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ELCoMoEDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emoji_text = row['emoji_text']  # 5 emoji descriptions joined
        sent2 = row['sent2']
        label = int(row['label'])

        encoding = self.tokenizer(
            emoji_text,
            sent2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =============================
# 2. MoE Model Definition
# =============================
import torch.nn as nn
from transformers import BertModel

class BertExpert(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output
        return self.classifier(self.dropout(cls))


class MoEForEntailment(nn.Module):
    def __init__(self, num_experts, model_path, num_labels):
        super().__init__()
        self.experts = nn.ModuleList([BertExpert(model_path, num_labels) for _ in range(num_experts)])
        self.router_bert = BertModel.from_pretrained(model_path)
        self.router = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask):
        router_output = self.router_bert(input_ids, attention_mask).pooler_output
        weights = self.router(router_output)  # (batch, num_experts)

        expert_logits = []
        for expert in self.experts:
            logits = expert(input_ids, attention_mask)  # (batch, num_labels)
            expert_logits.append(logits)

        expert_logits = torch.stack(expert_logits, dim=1)  # (batch, num_experts, num_labels)
        weights = weights.unsqueeze(-1)  # (batch, num_experts, 1)
        final_logits = torch.sum(expert_logits * weights, dim=1)  # (batch, num_labels)
        return final_logits


# =============================
# 3. Training Function
# =============================
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score

def train(model, train_loader, val_loader, device, save_path='best_model.pt', epochs=3, lr=2e-5):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        # Early stopping logic: save model if val improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")

    print(f"\nBest Val Acc = {best_val_acc:.4f} at Epoch {best_epoch}")

# def evaluate(model, dataloader, device):
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             logits = model(input_ids=input_ids, attention_mask=attention_mask)
#             preds = torch.argmax(logits, dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     acc = accuracy_score(all_labels, all_preds)
#     return acc