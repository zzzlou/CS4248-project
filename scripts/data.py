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
from tqdm import tqdm

model_name = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

path = "/home/gaobin/zzlou/folder/vlm/exp-entailment/test.csv"
data = pd.read_csv(path)

def extract_emoji_names(text):
    text = text.replace("This is ", "", 1)
    
    # split by "[EM] ""
    emoji_names = [part.strip() for part in text.split("[EM]") if part.strip()]

    emoji_names = [re.sub(r"[^\w\d_]", "", name) for name in emoji_names]
    
    return emoji_names if emoji_names else None

def get_unicode_representation(emoji_symbol):
    return '-'.join(f"{ord(c):X}" for c in emoji_symbol).lower()

def get_unicode_representation_upper(emoji_symbol):
    return ' '.join(f"U+{ord(c):X}" for c in emoji_symbol)

emoji_dir = "/home/gaobin/zzlou/folder/vlm/emojis"
def get_emoji_images(emoji_names):
    images = []
    
    for emoji_name in emoji_names:
        image_path = os.path.join(emoji_dir, f"{emoji_name}.png")

        # check if path exists
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
        else:
            print(f"⚠️ {emoji_name} Image Not Found {image_path}")
            images.append(None)  
    
    return images
    
def get_emoji_images_from_web(emoji_names):
    images = []
    
    for emoji_name in emoji_names:
        emoji_symbol = emoji.emojize(f":{emoji_name}:", language="alias")


        unicode = get_unicode_representation(emoji_symbol)
        upper_unicode = get_unicode_representation_upper(emoji_symbol)
        link = f"https://emoji.aranja.com/static/emoji-data/img-apple-160/{unicode}.png"
        
        response = requests.get(link)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            images.append(image)
            save_path = os.path.join("/home/gaobin/zzlou/folder/vlm/emojis", f"{emoji_name}.png")
            image.save(save_path, format="PNG")

        else:
            images.append(None)  

    
    return images 
    
def generate_description(image):
    if image is None:
        return "❌ Image not found"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give 5 different detailed descriptions of the image"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
    generate_ids = model.generate(**inputs, max_new_tokens=500)[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

new_data = []


for idx, row in tqdm(data.iterrows(),total = len(data), desc="Generating Descriptions"):
    emoji_names = extract_emoji_names(row["sent1"])  
    if emoji_names:
        images = get_emoji_images(emoji_names)
        descriptions = []
        for image in images:
            descriptions.append(generate_description(image))
    else:
        descriptions, image_url = "❌ No emojis found", None
        print(row)

    new_data.append({
        "sent1": row["sent1"],
        "sent2": row["sent2"],
        "generated_description": descriptions,
        "strategy": row["strategy"],
        "label": row["label"]
    })

df_new = pd.DataFrame(new_data)

# save as CSV
output_path = "test.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(new_data, json_file, ensure_ascii=False, indent=4)

print(f"✅ 结果已保存到 {output_path}")