import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, random_split

from data_prep import EmojiMoEDataset
from model import MoEModel  # 假设你模型写在 moe_model.py 里
from model import train   # 假设训练代码也写在单独文件里
from transformers import logging
logging.set_verbosity_error()
def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Load tokenizer and dataset
    model_path = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    data_path = "/home/gaobin/zzlou/folder/train.json"
    dataset = EmojiMoEDataset(data_path, tokenizer, max_length=128, desc_limit=2)

    # 3. Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    model = MoEModel(
        num_experts=5,
        model_path=model_path,
        num_labels=2  # binary classification
    )

    # 5. Start training
    print("training starts")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path='best_moe_model.pt',
        epochs=5,
        lr=2e-5
    )

if __name__ == "__main__":
    main()
