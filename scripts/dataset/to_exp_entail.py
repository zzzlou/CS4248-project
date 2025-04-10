import json
import ast
import pandas as pd
import sklearn
import emoji


# read descriptions
descriptions_path = "../../data/prepare/descriptions_mapping.json"
with open(descriptions_path, "r") as f:
    descriptions_mapping = json.load(f)


# read entailment dataset
ELCo_path = "../../data/prepare/final_ELCo.csv"
ELCo_dataset = pd.read_csv(ELCo_path)

# Initialize list to store all examples
dataset = []


# Process each row in ELCo dataset
for _, row in ELCo_dataset.iterrows():

    # Get descriptions for each emoji
    descriptions = []
    emoji_dict = emoji.emoji_list(row['EM'])
    emojis = [emoji['emoji'] for emoji in emoji_dict]
    for emoji_unique in emojis:
        if emoji_unique in descriptions_mapping:
            descriptions.append(descriptions_mapping[f"{emoji_unique}"])
        else:
            print(f"Emoji {emoji_unique} not found in descriptions_mapping")
            print(f"Row: {row['EM']}")
            print(f"{emoji_unique}")
    
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
output_path = "../../data/prepare/dataset_exp_entail.json"
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=2)



test_path = "../../data/test.json"
train_path = "../../data/train.json"
val_path = "../../data/val.json"

train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size=0.1, random_state=42)
train_dataset, val_dataset = sklearn.model_selection.train_test_split(train_dataset, test_size=1/9, random_state=42)

with open(test_path, 'w') as f:
    json.dump(test_dataset, f, indent=2)

with open(train_path, 'w') as f:
    json.dump(train_dataset, f, indent=2)

with open(val_path, 'w') as f:
    json.dump(val_dataset, f, indent=2)





def to_uXXXX(s: str) -> str:
    """
    Return a string that represents each character in s as:
    - \\uXXXX for BMP codepoints, or
    - \\uXXXX\\uXXXX for codepoints above U+FFFF (using surrogate pairs).
    """
    result = []
    for ch in s:
        cp = ord(ch)
        if cp <= 0xFFFF:
            # Within BMP
            result.append(f'\\u{cp:04x}')
        else:
            # Above BMP, so convert to surrogate pair
            cp -= 0x10000
            high_surrogate = 0xD800 + (cp >> 10)
            low_surrogate  = 0xDC00 + (cp & 0x3FF)
            result.append(f'\\u{high_surrogate:04x}\\u{low_surrogate:04x}')
    return ''.join(result)
