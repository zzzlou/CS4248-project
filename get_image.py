import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import emoji
import sys
from io import BytesIO
import re
import json
import os

path = "exp-entailment/train.csv"
data = pd.read_csv(path)

def extract_emoji_names(text):
    # remove "This is "
    text = text.replace("This is ", "", 1)
    
    # split by [EM]
    emoji_names = [part.strip() for part in text.split("[EM]") if part.strip()]

    emoji_names = [re.sub(r"[^\w\d_]", "", name) for name in emoji_names]
    
    return emoji_names if emoji_names else None

def get_unicode_representation(emoji_symbol):
    return '-'.join(f"{ord(c):X}" for c in emoji_symbol).lower()

def get_emoji_images(emoji_names):
    images = []
    
    for emoji_name in emoji_names:
        emoji_symbol = emoji.emojize(f":{emoji_name}:", language="alias")


        unicode = get_unicode_representation(emoji_symbol)
        link = f"   "
        
        response = requests.get(link)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            images.append(image)
            save_path = os.path.join("/home/gaobin/zzlou/folder/vlm/emojis", f"{emoji_name}.png")
            image.save(save_path, format="PNG")

        else:
            print(emoji_name)
            images.append(None)  # None if download failed

    
    return images 
    



for idx, row in data.iloc[739:].iterrows():
    emoji_names = extract_emoji_names(row["sent1"])  
    
    if emoji_names:
        images = get_emoji_images(emoji_names)
        print(f"{idx} success")
    else:
        print(f"{idx} failed")