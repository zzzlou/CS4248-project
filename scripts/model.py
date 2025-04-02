# =============================
# 1. Data Preparation
# =============================
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from collections import Counter, defaultdict
import torch.nn.functional as F



# =============================
# 2. MoE Model Definition with LoRA Experts
# =============================
import torch.nn as nn
from transformers import BertModel
from peft import get_peft_model, LoraConfig, TaskType

from peft import PeftModelForSequenceClassification
from transformers import BertForSequenceClassification

class BertExpert(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        base_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        self.bert = get_peft_model(base_model, lora_config)

    def forward(self, input_ids, attention_mask,**kwargs):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits


class DualHeadRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_error_types):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.dispatch_head = nn.Linear(hidden_dim, num_experts)   # for expert routing
        self.error_type_head = nn.Linear(hidden_dim, num_error_types)  # for error type prediction

    def forward(self, x):
        # x: (batch, input_dim)  here x comes from BERT's pooler_output [CLS]
        h = F.relu(self.hidden_layer(x))
        # Routing weights: softmax over experts
        g = F.softmax(self.dispatch_head(h), dim=-1)  # (batch, num_experts)
        # Error type prediction logits 
        p = self.error_type_head(h)  # (batch, num_error_types)
        return g, p
    
class MoEModel(nn.Module):
    def __init__(self, num_experts, model_path, num_labels,num_error_types):
        super().__init__()
        self.experts = nn.ModuleList([BertExpert(model_path, num_labels) for _ in range(num_experts)])
        # extract text features using bert encoder, then pass as input to router
        self.router_bert = BertModel.from_pretrained(model_path)
        self.router = DualHeadRouter(input_dim=768, hidden_dim=128, num_experts=num_experts, num_error_types=num_error_types)

    def forward(self, input_ids, attention_mask, return_router=False):
        router_out = self.router_bert(input_ids, attention_mask).pooler_output
        g, p = self.router(router_out) # g: (batch, num_experts), p: (batch, num_error_types)

        # calculate all expert outputs
        expert_logits = [expert(input_ids, attention_mask) for expert in self.experts]
        expert_logits = torch.stack(expert_logits, dim=1)  # (batch, num_experts, num_labels)
        # aggregate
        g_unsq = g.unsqueeze(-1)  # (batch, num_experts, 1)
        final_logits = torch.sum(expert_logits * g_unsq, dim=1)  # (batch, num_labels)
        
        if return_router:
            return final_logits, g, p
        return final_logits

# =============================
# 3. Training Function
# =============================
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score

def load_balance_loss(g_weights):
    # g_weights: (batch, num_experts)
    mean_prob = g_weights.mean(dim=0)  # (num_experts,)
    num_experts = g_weights.size(1)
    loss = (mean_prob ** 2).sum() * (num_experts ** 2)
    return loss

def error_type_loss(p_logits, gold_labels):
    # p_logits: (batch, num_error_types); gold_labels: (batch,)
    return F.cross_entropy(p_logits, gold_labels)

def train(model, train_loader, val_loader, device, num_experts, alpha=1.0, beta=0.01, save_path='best_model.pt', epochs=3, lr=2e-5):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    #  loss: sentence entailment classification loss
    criterion_cls = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        total_cls_loss = 0
        total_type_loss = 0
        total_balance_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # strategy is used for error type label
            type_labels = batch['strategy'].to(device)

            optimizer.zero_grad()
            final_logits, g_weights, p_logits = model(input_ids, attention_mask, return_router=True)
            
            loss_cls = criterion_cls(final_logits, labels)
            loss_type = error_type_loss(p_logits, type_labels)
            loss_balance = load_balance_loss(g_weights)
            
            loss = loss_cls + alpha * loss_type + beta * loss_balance
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_type_loss += loss_type.item()
            total_balance_loss += loss_balance.item()

            preds = torch.argmax(final_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f} (Cls: {total_cls_loss:.4f}, Type: {total_type_loss:.4f}, Bal: {total_balance_loss:.4f}) - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        analyze_router_distribution(model, train_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")

    print(f"\nBest Val Acc = {best_val_acc:.4f} at Epoch {best_epoch}")

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

def analyze_router_distribution(model, dataloader, device):
    model.eval()
    argmax_counts = Counter()
    softmax_sums = defaultdict(float)
    num_total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing router"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            router_out = model.router_bert(input_ids, attention_mask).pooler_output
            g_weights, _ = model.router(router_out)
            top_expert = torch.argmax(g_weights, dim=1)
            for idx in top_expert.cpu().tolist():
                argmax_counts[idx] += 1
            weights_cpu = g_weights.cpu()
            for i in range(weights_cpu.size(0)):
                for expert_id, weight in enumerate(weights_cpu[i]):
                    softmax_sums[expert_id] += weight.item()
            num_total_samples += weights_cpu.size(0)
    print("🔍 Router Argmax Distribution (Hard Routing):")
    for expert_id in sorted(argmax_counts):
        count = argmax_counts[expert_id]
        print(f"Expert {expert_id}: {count} samples ({count / num_total_samples:.2%})")
    print("\n📊 Router Softmax Weights Distribution (Soft Routing):")
    for expert_id in sorted(softmax_sums):
        avg_weight = softmax_sums[expert_id] / num_total_samples
        print(f"Expert {expert_id}: average weight = {avg_weight:.4f}")
    return argmax_counts, softmax_sums