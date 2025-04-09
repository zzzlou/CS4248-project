import json
import ast
import pandas as pd
import sklearn


# read descriptions
descriptions_path = "/workspace/CS4248-project/data/prepare/descriptions_mapping.json"
with open(descriptions_path, "r") as f:
    descriptions_mapping = json.load(f)


# read entailment dataset
ELCo_path = "/workspace/CS4248-project/data/prepare/final_ELCo.csv"
ELCo_dataset = pd.read_csv(ELCo_path)

# Initialize list to store all examples
dataset = []

# Process each row in ELCo dataset
for _, row in ELCo_dataset.iterrows():

    # Get descriptions for each emoji
    descriptions = []
    for emoji in row['EM']:
        if emoji in descriptions_mapping:
            descriptions.append(descriptions_mapping[emoji])
    
    # Create example dictionary
    example = {
        'sent1': f"This is {row['Description']}", 
        'sent2': f"This is {row['EN']}",
        'strategy': row['Composition strategy'],
        'label': 0 if row['Attribute'] == "Negative_Random" else 1,
        'generated_description': descriptions
    }
    
    dataset.append(example)

# Save to JSON file
output_path = "/workspace/CS4248-project/data/prepare/dataset_exp_entail.json"
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=2)



test_path = "/workspace/CS4248-project/data/test.json"
train_path = "/workspace/CS4248-project/data/train.json"
val_path = "/workspace/CS4248-project/data/val.json"

train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size=0.1, random_state=42)
train_dataset, val_dataset = sklearn.model_selection.train_test_split(train_dataset, test_size=1/9, random_state=42)

with open(test_path, 'w') as f:
    json.dump(test_dataset, f, indent=2)

with open(train_path, 'w') as f:
    json.dump(train_dataset, f, indent=2)

with open(val_path, 'w') as f:
    json.dump(val_dataset, f, indent=2)





