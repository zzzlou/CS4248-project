import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import logging
logging.set_verbosity_error()

# ---------------------------
# 1. Data Preparation 
# ---------------------------
class EmojiMoEDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        import json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # concatenate all descriptions for emojis in the same sentence
        desc = "[EM]".join(item['processed_generated_description'])
        sent2 = item['sent2']
        label = int(item['label'])
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}
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
            # 如果需要用来统计策略，这里保留也行
            'strategy': torch.tensor(strategy, dtype=torch.long)
        }

# ---------------------------
# 2. Pure BERT Model Definition (Basic fine-tuning version)
# ---------------------------
class PureBERTModel(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        # 直接加载 BertForSequenceClassification
        self.bert = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

# ---------------------------
# 3. Training & Evaluation Functions
# ---------------------------
def train(model, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-5, save_path='best_purebert.pt'):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            strategies = batch['strategy']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f}")
        
        # Evaluate on validation set: overall and per-strategy
        val_acc, val_strategy_accuracy = evaluate(model, val_loader, device, per_strategy=True)
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")
        print("Per-strategy validation accuracy:")
        for strat, (correct, total, acc) in sorted(val_strategy_accuracy.items()):
            print(f"  Strategy {strat}: {acc:.1f}% ({correct}/{total})")
        
        # Evaluate on test set as well
        test_acc, test_strategy_accuracy = evaluate(model, test_loader, device, per_strategy=True)
        print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.4f}")
        print("Per-strategy test accuracy:")
        for strat, (correct, total, acc) in sorted(test_strategy_accuracy.items()):
            print(f"  Strategy {strat}: {acc:.1f}% ({correct}/{total})")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")

    print(f"Best Val Acc = {best_val_acc:.4f} at Epoch {best_epoch}")
    

def evaluate(model, dataloader, device, per_strategy=False):
    model.eval()
    all_preds, all_labels, all_strategies = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            # strategy 用于统计，这里仍保留到 CPU 上
            strategies = batch['strategy']
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_strategies.extend(strategies.cpu().numpy())
    
    overall_acc = accuracy_score(all_labels, all_preds)
    per_strategy_accuracy = {}
    if per_strategy:
        strategy_counts = defaultdict(lambda: {"correct": 0, "total": 0})
        for pred, true, strat in zip(all_preds, all_labels, all_strategies):
            strategy_counts[strat]["total"] += 1
            if pred == true:
                strategy_counts[strat]["correct"] += 1
        per_strategy_accuracy = {
            strat: (counts["correct"], counts["total"], round(100.0 * counts["correct"] / counts["total"], 1))
            for strat, counts in strategy_counts.items() if counts["total"] > 0
        }
    return overall_acc, per_strategy_accuracy

# ---------------------------
# 4. Main Function: Load data, model, start training
# ---------------------------
def main():
    # model_path = "bert-base-uncased"
    model_path = "textattack/bert-base-uncased-mnli"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # tokenizer.add_tokens(['[EM]'])
    
    # 注意数据文件路径需要你自己修改
    train_json = "/home/gaobin/zzlou/folder/vlm/data/fixed_train.json"
    val_json = "/home/gaobin/zzlou/folder/vlm/data/fixed_val.json"
    test_json = "/home/gaobin/zzlou/folder/vlm/data/fixed_test.json"

    # train_json = "/home/gaobin/zzlou/folder/vlm/exp-entailment/train.csv"
    # val_json = "/home/gaobin/zzlou/folder/vlm/exp-entailment/val.csv"
    # test_json = "/hom选·行·行·e/gaobin/zzlou/folder/vlm/exp-entailment/test.csv"
    
    train_dataset = EmojiMoEDataset(train_json, tokenizer, max_length=256)
    val_dataset = EmojiMoEDataset(val_json, tokenizer, max_length=256)
    test_dataset = EmojiMoEDataset(test_json, tokenizer, max_length=256)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_labels = 2 
    model = PureBERTModel(model_path, num_labels=num_labels)
    # model.bert.resize_token_embeddings(len(tokenizer))
    
    train(model, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-5, save_path='best_desc_purebert4-11-1.pt')
    
if __name__ == "__main__":
    main()