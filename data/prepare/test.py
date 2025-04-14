import csv
import emoji

# Read emojis from CSV file
emojis = []
with open('unique_emojis.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0]:  # Check if row exists and is not empty
            emojis.append(emoji.emoji_list(row[0])[0]['emoji'])

print(f"emojis: {emojis}")
