from collections import defaultdict
import random
import json
from collections import Counter

def balance_by_strategy(json_path, strategy_key='strategy', seed=42):
    random.seed(seed)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # group by strategy
    buckets = defaultdict(list)
    for item in data:
        strategy = item.get(strategy_key)
        if strategy is None:
            raise ValueError(f"Missing `{strategy_key}` in data sample: {item}")
        buckets[strategy].append(item)

    # find the class with least amount of samples
    min_count = min(len(items) for items in buckets.values())
    print(f"🧮 Strategy distribution before balancing:")
    for k, v in buckets.items():
        print(f"  Strategy {k}: {len(v)} samples")

    # sample min_count samples from every class
    balanced_data = []
    for strategy, items in buckets.items():
        sampled = random.sample(items, min_count)
        balanced_data.extend(sampled)

    random.shuffle(balanced_data)

    print(f"✅ Balanced dataset contains {len(balanced_data)} samples, {min_count} per strategy")
    return balanced_data

def balance_with_upsample(json_path, strategy_key='strategy', target_count=300, seed=42):
    random.seed(seed)
    with open(json_path, 'r') as f:
        data = json.load(f)

    buckets = defaultdict(list)
    for item in data:
        strategy = item.get(strategy_key)
        if strategy is None:
            raise ValueError(f"Missing `{strategy_key}` in data sample: {item}")
        buckets[strategy].append(item)

    print("🧮 Strategy distribution before balancing:")
    for k, v in buckets.items():
        print(f"  Strategy {k}: {len(v)} samples")

    balanced_data = []
    for strategy, items in buckets.items():
        if len(items) >= target_count:
            sampled = random.sample(items, target_count)
        else:
            sampled = random.choices(items, k=target_count)
        balanced_data.extend(sampled)

    random.shuffle(balanced_data)
    print(f"✅ Balanced (upsampled) dataset contains {len(balanced_data)} samples, ~{target_count} per strategy")
    return balanced_data

balanced_data = balance_with_upsample("/home/gaobin/zzlou/folder/train.json")
with open("/home/gaobin/zzlou/folder/balanced_train.json", "w") as f:
    json.dump(balanced_data, f, indent=2)



with open("/home/gaobin/zzlou/folder/balanced_train.json", "r") as f:
    balanced_data = json.load(f)

strategy_counter = Counter(item["strategy"] for item in balanced_data)

print("📊 Strategy distribution after balancing:")
for strategy, count in sorted(strategy_counter.items()):
    print(f"  Strategy {strategy}: {count} samples")