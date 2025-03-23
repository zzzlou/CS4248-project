import torch
import emoji
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

### **Step 1: enter emoji name & target sentence**
emoji_name = "necktie"
sentence2 = "This is big business."

### **Step 2: get Emoji Unicode**
emoji_symbol = emoji.emojize(f":{emoji_name}:")
unicode_code = f"{ord(emoji_symbol):X}".lower()

### **Step 3: download Emoji png**
link = f"https://emoji.aranja.com/static/emoji-data/img-apple-160/{unicode_code}.png"
response = requests.get(link)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))  
    print("✅ Emoji Image download success")
else:
    print("❌ Image download failed:", response.status_code)
    exit()


device = "cuda" if torch.cuda.is_available() else "cpu"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},  
            {"type": "text", "text": "Give 5 different detailed descriptions of the image"},
        ],
    },
]
model_name = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


generate_ids = model.generate(**inputs, max_new_tokens=500)
descriptions = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]


print("\n✅ LLaVA generate description:\n", descriptions)

# decide if entail or not
llama_model_id = "meta-llama/Llama-3.2-3B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, device_map="auto")


llama_prompt = f"""Given several text descriptions of an image, and a sentence, determine whether the meaning of the sentence is entailed in the text descriptions. 
Consider indirect implications and contextual meaning, not just exact word matches. 
If the descriptions may imply the sentence's meaning, output 'Yes'. Otherwise, output 'No'. 
Strictly only output 'Yes' or 'No', no explanation is needed.

Descriptions:
{descriptions}

Sentence: {sentence2}
"""

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": llama_prompt}
]


formatted_input = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = llama_tokenizer(formatted_input, return_tensors="pt").input_ids.to(llama_model.device)

output_ids = llama_model.generate(input_ids, max_new_tokens=100)[:, input_ids.shape[1]:]
output_text = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n✅ LLaMA Result:", output_text)
