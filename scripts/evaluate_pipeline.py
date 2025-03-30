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
from model import MoEModel
from transformers import BertTokenizer



def generate_description(image,model,processor):
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

def extract_emoji_names(text):
    # 去掉 "This is "（如果存在的话）
    text = text.replace("This is ", "", 1)
    
    # 使用 [EM] 进行分割
    emoji_names = [part.strip() for part in text.split("[EM]") if part.strip()]

    emoji_names = [re.sub(r"[^\w\d_]", "", name) for name in emoji_names]
    
    return emoji_names if emoji_names else None

def get_emoji_images(emoji_names):
    images = []
    
    for emoji_name in emoji_names:
        image_path = os.path.join("/home/gaobin/zzlou/folder/vlm/emojis", f"{emoji_name}.png")

        # 检查文件是否存在
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
        else:
            print(f"⚠️ {emoji_name} Image not found: {image_path}")
            images.append(None)  # 图片不存在，填充 None
    
    return images
    
def evaluate(data, vlm_model, moe_model, tokenizer, processor, device):
    """
    对整个测试集进行端到端评估：
      - 使用 VL 模型在线生成每个样本的描述
      - 用生成的描述作为输入，经 tokenizer 编码后送入 MoE 模型预测标签

    Args:
        data (pd.DataFrame): 测试集数据，至少包含 "sent1", "sent2", "strategy", "label"
        vlm_model: VL 模型，用于生成描述（例如 LLaVA）
        moe_model: MoE 模型，用于判断文本对是否蕴含（分类模型）
        tokenizer: 用于 MoE 模型的 tokenizer
        processor: 用于 VL 模型的 processor（例如 AutoProcessor）
        device: 设备（cuda 或 cpu）

    Returns:
        results (list): 每个样本的详细信息，包括生成的描述、真实标签与预测标签
    """
    moe_model.eval()  # 切换到评估模式
    results = []

    # 不需要梯度计算
    with torch.no_grad():
        for idx, row in data.iterrows():
            # 提取 emoji 名称
            emoji_names = extract_emoji_names(row["sent1"])
            # 根据 emoji 名称获取图像列表
            images = get_emoji_images(emoji_names) if emoji_names else []
            descriptions = []
            # 对每个图像生成描述；注意这里可能耗时较长
            for image in images:
                desc = generate_description(image, vlm_model, processor)
                descriptions.append(desc)
            # 将多个描述拼接为一个文本输入
            combined_desc = " ".join(descriptions)
            # 对文本进行编码；注意这里设定了 max_length，需要根据实际情况调整
            inputs = tokenizer(
                combined_desc,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            # 将所有张量转移到目标设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # 使用 MoE 模型预测
            logits = moe_model(**inputs)
            pred = torch.argmax(logits, dim=1).item()

            # 保存结果
            results.append({
                "sent1": row["sent1"],
                "sent2": row["sent2"],
                "generated_description": descriptions,
                "strategy": row["strategy"],
                "label": row["label"],
                "prediction": pred
            })
    return results


        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 VL 模型和 processor (例如 LLaVA)
    vlm_model_name = "llava-hf/llava-1.5-7b-hf"
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        vlm_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    
    # 2. 加载测试集数据
    test_csv_path = "/home/gaobin/zzlou/folder/vlm/exp-entailment/test.csv"
    data = pd.read_csv(test_csv_path)
    
    # 3. 加载 MoE 模型
    # 假设 MoE 模型是基于 "bert-base-uncased"
    moe_model = MoEModel(num_experts=5, model_path="bert-base-uncased", num_labels=2)
    moe_model_checkpoint = "/home/gaobin/zzlou/folder/vlm/exp-entailment/best_moe_model.pt"
    moe_model.load_state_dict(torch.load(moe_model_checkpoint, map_location=device))
    moe_model.to(device)
    moe_model.eval()
    
    # 4. 加载 tokenizer (例如使用 bert-base-uncased 对应的 tokenizer)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 5. 评估整个测试集
    results = evaluate(data, vlm_model, moe_model, tokenizer, processor, device)
    
    # 6. 保存结果到 JSON 文件
    output_path = "/home/gaobin/zzlou/folder/vlm/exp-entailment/test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("Evaluation completed. Results saved to:", output_path)


    

