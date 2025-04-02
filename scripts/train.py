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
    data_path = "/home/gaobin/zzlou/folder/balanced_train.json"
    dataset = EmojiMoEDataset(data_path, tokenizer, max_length=256, desc_limit=3)

    # 3. Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # 修改点1：增加 num_error_types 参数，例如设置为7
    model = MoEModel(
        num_experts=6,
        model_path=model_path,
        num_labels=2,          # binary classification
        num_error_types=6      # 根据数据中 strategy 的类别数来设置
    )

    # 5. Start training
    print("training starts")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_experts=6,         # 传入 expert 数量，用于负载均衡 loss 的计算
        alpha=1.0,             # error type loss 权重
        beta=0.01,             # load balancing loss 权重
        save_path='best_moe_model.pt',
        epochs=20,
        lr=2e-5
    )

if __name__ == "__main__":
    main()