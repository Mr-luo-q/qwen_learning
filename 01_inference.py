import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": "请用简单的话解释什么是强化学习。"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

print("开始生成...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\n===== 模型回答 =====")
print(response)