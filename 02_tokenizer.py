# 02_tokenizer.py

from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {
        "role": "system",
        "content": "你是一个强化学习老师。"
    },
    {
        "role": "user",
        "content": "PPO是什么？"
    },
    {
        "role": "assistant",
        "content": "PPO是一种策略优化强化学习算法。"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print("===== Chat Template 后 =====")
print(text)

tokens = tokenizer(
    text,
    return_tensors="pt"
)

print("\n===== Token IDs =====")
print(tokens["input_ids"])