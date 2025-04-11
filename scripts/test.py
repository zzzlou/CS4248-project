import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel, CLIPFeatureExtractor
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict
from transformers import logging
from PIL import Image
import os
import re
from transformers import CLIPImageProcessor

logging.set_verbosity_error()


def extract_emoji_names(text):
    # Remove prefix if present
    text = text.replace("This is ", "", 1)
    # Split by "[EM]"
    emoji_names = [part.strip() for part in text.split("[EM]") if part.strip()]
    # Remove non-word, digit, or underscore characters
    emoji_names = [re.sub(r"[^\w\d_]", "", name) for name in emoji_names]
    return emoji_names


def get_emoji_images(emoji_names):
    images = []
    emoji_dir = "/home/gaobin/zzlou/folder/vlm/emojis"
    for emoji_name in emoji_names:
        image_path = os.path.join(emoji_dir, f"{emoji_name}.png")
        if os.path.exists(image_path):
            # Ensure image is opened in RGB mode
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        else:
            print(f"⚠️ {emoji_name} Image Not Found: {image_path}")
    return images


class EmojiMoEDataset(Dataset):
    def __init__(self, csv_path, tokenizer, feature_extractor, clip_vision, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor  # CLIP feature extractor
        self.clip_vision = clip_vision              # Preloaded CLIP vision model
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].to_dict()
        sent1 = item["sent1"]
        sent2 = item["sent2"]
        label = int(item["label"])
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}
        strategy = mapping[int(item["strategy"])]

        # Encode text (squeeze to remove extra batch dimension)
        encoded = self.tokenizer(
            sent1, sent2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Extract emoji names and load corresponding images
        emoji_names = extract_emoji_names(sent1)
        images_list = get_emoji_images(emoji_names) if emoji_names else []
        
        images_list = [img for img in images_list if img is not None]
        if images_list:
            # Extract CLIP pixel tensors
            pixel_values = self.feature_extractor(images=images_list, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                clip_outputs = self.clip_vision(pixel_values=pixel_values)
            # Average the pooled outputs of all images
            visual_emb = clip_outputs.pooler_output.mean(dim=0)  # [clip_hidden_size]
        else:
            # Return a zero vector if no image found (should not happen in normal cases)
            visual_emb = torch.zeros(self.clip_vision.config.hidden_size)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),          # [seq_len]
            'attention_mask': encoded['attention_mask'].squeeze(0),    # [seq_len]
            'token_type_ids': encoded['token_type_ids'].squeeze(0),    # [seq_len]
            'labels': torch.tensor(label, dtype=torch.long),
            'strategy': torch.tensor(strategy, dtype=torch.long),
            'images': visual_emb  # Visual feature vector [clip_hidden_size]
        }


class EndToEndVisualBERT(nn.Module):
    def __init__(self, bert_model_path, clip_hidden_size, num_labels, hidden_dim=768):
        """
        该模型将 BERT 的 [CLS] 向量和经过视觉映射的 emoji 特征拼接后进入分类 MLP
        """
        super().__init__()
        # Text branch: load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_path)
        # Visual projection layer: map CLIP features to BERT hidden space
        self.visual_proj = nn.Linear(clip_hidden_size, hidden_dim)
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, images):
        # Text branch: get [CLS] representation
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        cls_text = bert_outputs.pooler_output  # [batch, hidden_dim]

        # Project visual features to match BERT space
        mapped_visual = self.visual_proj(images)  # [batch, hidden_dim]

        # Concatenate and classify
        fused_repr = torch.cat([cls_text, mapped_visual], dim=1)  # [batch, hidden_dim*2]
        logits = self.classifier(fused_repr)
        return logits


def train(model, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-5, save_path='best_model.pt'):
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
            images = batch['images'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f}")

        # Evaluate on validation set
        val_acc, val_strategy_accuracy = evaluate(model, val_loader, device, per_strategy=True)
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f}")
        print("Per-strategy validation accuracy:")
        for strat, (correct, total, acc) in sorted(val_strategy_accuracy.items()):
            print(f"  Strategy {strat}: {acc:.1f}% ({correct}/{total})")

        # Evaluate on test set
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
            strategies = batch['strategy']
            images = batch['images'].to(device)
            logits = model(input_ids, attention_mask, token_type_ids, images)
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


def main():
    bert_model_path = "textattack/bert-base-uncased-mnli"
    clip_model_name = "openai/clip-vit-base-patch32"

    # Load CLIP vision model and freeze it
    clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
    clip_vision.eval()
    clip_hidden_size = clip_vision.config.hidden_size

    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    tokenizer.add_tokens(['[EM]'])
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_name)

    
    train_csv = "/home/gaobin/zzlou/folder/vlm/exp-entailment/train.csv"
    val_csv = "/home/gaobin/zzlou/folder/vlm/exp-entailment/val.csv"
    test_csv = "/home/gaobin/zzlou/folder/vlm/exp-entailment/test.csv"

    # Construct datasets
    train_dataset  = EmojiMoEDataset(train_csv, tokenizer, feature_extractor, clip_vision, max_length=256)
    val_dataset = EmojiMoEDataset(val_csv, tokenizer, feature_extractor, clip_vision, max_length=256)
    test_dataset = EmojiMoEDataset(test_csv, tokenizer, feature_extractor, clip_vision, max_length=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_labels = 2  
    model = EndToEndVisualBERT(bert_model_path, clip_hidden_size, num_labels)
    # Resize BERT embeddings to account for new “[EM]” token
    model.bert.resize_token_embeddings(len(tokenizer))
    
    train(model, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-5, save_path='best_combbert_4_11_1.pt')


if __name__ == "__main__":
    main()
