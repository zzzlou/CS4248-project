import re
import csv

log_path = "train_1.log"          
output_csv = "router_softmax.csv" 

expert_weights_per_epoch = []

with open(log_path, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if "Router Softmax Weights Distribution" in lines[i]:
        current_weights = []

        for j in range(i+1, i+7):
            match = re.search(r"Expert\s+(\d+):\s+average weight\s*=\s*([\d\.]+)", lines[j])
            if match:
                weight = float(match.group(2))
                current_weights.append(weight)
        if len(current_weights) == 6:
            expert_weights_per_epoch.append(current_weights)
        i += 7  
    else:
        i += 1


with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"Expert_{i}" for i in range(6)])
    writer.writerows(expert_weights_per_epoch)

print(f"✅ Parsed {len(expert_weights_per_epoch)} epochs. Saved to {output_csv}.")
