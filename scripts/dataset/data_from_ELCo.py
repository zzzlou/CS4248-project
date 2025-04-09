import pandas as pd
import random
from wonderwords import RandomWord

random_word = RandomWord()
sample_size = 400
# Read the CSV file
df = pd.read_csv('data/elco/upsampled_ELCo.csv')

# Get unique composition strategies
strategies = df['Composition strategy'].unique()

# Initialize dictionary to store sampled entries
sampled_data = {}
negative_data = {}
# Sample 50 entries for each strategy
for strategy in strategies:
    strategy_df = df[df['Composition strategy'] == strategy]
    if len(strategy_df) >= sample_size:
        sampled_data[strategy] = strategy_df.sample(n=sample_size, random_state=42)
    else:
        # If less than 50 entries, take all available
        sampled_data[strategy] = strategy_df

    # generate sample_size random phrases and replace the original phrase with the random phrase
    random_phrases = pd.DataFrame(columns=['phrase'])
    for i in range(sample_size):
        adj = random_word.word(include_parts_of_speech=["adjectives"])
        noun = random_word.word(include_parts_of_speech=["nouns"])
        random_phrases.loc[i] = f"{adj} {noun}"
    base = sampled_data[strategy].copy(deep=True)  # Use deepcopy to avoid modifying source data
    base['EN'] = random_phrases['phrase'].values  # Convert to numpy array to ensure proper assignment
    base['Attribute'] = "Negative_Random"
    negative_data[strategy] = base

# Combine all sampled data
positive_df = pd.concat(sampled_data.values())
negative_df = pd.concat(negative_data.values())
final_df = pd.concat([positive_df, negative_df])


# # Save to new CSV file
final_df.to_csv('data/elco/final_ELCo.csv', index=False)

