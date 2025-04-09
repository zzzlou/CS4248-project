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

"""
Generate dataset for MoE training from the original dataset used in ELCo and emoji images from web
Input:
    - images of each uniqueemoji
    - a VLM model
Output:
    - dataset_name.json: generated dataset ready for our task
"""


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


if __name__ == "__main__":

    # load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # load images
    image_dir = "/workspace/CS4248-project/data/emojis"

    # output path
    output_path = "/workspace/CS4248-project/data/descriptions.json"

    # {emoji_unicode: [description1, description2, ...]}
    descriptions_mapping = {}
    generated = []
    for image_name in tqdm(os.listdir(image_dir), desc="Generating descriptions"):
        if image_name in generated:
            continue
        image_path = os.path.join(image_dir, f"{image_name}")
        image = Image.open(image_path)
        descriptions_mapping[image_name[0:-4]] = generate_description(image)
        generated.append(image_name)
    # save as JSON
    with open(output_path, "w") as f:
        json.dump(descriptions_mapping, f)


