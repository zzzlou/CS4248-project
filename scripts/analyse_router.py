import torch
from collections import Counter
from tqdm import tqdm
from model import MoEModel
from data_prep import EmojiMoEDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from transformers import logging
logging.set_verbosity_error()

def analyze_router_distribution(model, dataloader, device):
    model.eval()
    argmax_counts = Counter()
    softmax_sums = defaultdict(float)  # expert_id -> accumulated softmax weight
    num_total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing router"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            router_output = model.router_bert(input_ids, attention_mask).pooler_output
            weights = model.router(router_output)  # shape: (batch_size, num_experts)

            # Argmax stats (hard routing)
            top_expert = torch.argmax(weights, dim=1)  # shape: (batch,)
            for idx in top_expert.cpu().tolist():
                argmax_counts[idx] += 1

            # Softmax stats (soft routing)
            weights_cpu = weights.cpu()
            for i in range(weights_cpu.size(0)):
                for expert_id, weight in enumerate(weights_cpu[i]):
                    softmax_sums[expert_id] += weight.item()

            num_total_samples += weights_cpu.size(0)

    print("🔍 Router Argmax Selection Distribution (Hard Routing):")
    for expert_id in sorted(argmax_counts):
        count = argmax_counts[expert_id]
        print(f"Expert {expert_id}: {count} samples ({count / num_total_samples:.2%})")

    print("\n📊 Router Softmax Weights Distribution (Soft Routing):")
    for expert_id in sorted(softmax_sums):
        avg_weight = softmax_sums[expert_id] / num_total_samples
        print(f"Expert {expert_id}: average weight = {avg_weight:.4f}")

    return argmax_counts, softmax_sums
import matplotlib.pyplot as plt

def plot_router_distribution(route_counts):
    expert_ids = list(sorted(route_counts.keys()))
    counts = [route_counts[i] for i in expert_ids]

    plt.figure(figsize=(8, 5))
    plt.bar(expert_ids, counts)
    plt.xlabel("Expert ID")
    plt.ylabel("Number of times selected (argmax)")
    plt.title("Router Expert Selection Distribution")
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
moe_model = MoEModel(num_experts=5, model_path="bert-base-uncased", num_labels=2)
moe_model_checkpoint = "/home/gaobin/zzlou/folder/best_moe_model_1.pt"
moe_model.load_state_dict(torch.load(moe_model_checkpoint, map_location=device))
moe_model.to(device)
moe_model.eval()

# use val_loader or train_loader
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
data_path = "/home/gaobin/zzlou/folder/train.json"
dataset = EmojiMoEDataset(data_path, tokenizer, max_length=128, desc_limit=2)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
analyze_router_distribution(moe_model, train_loader, device)