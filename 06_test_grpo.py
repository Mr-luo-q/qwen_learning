# 06_test_grpo.py
# 加载 GRPO 训练出的 LoRA 适配器并测试（对应 04_test_lora.py 的 GRPO 版）
# 注意：system 提示语必须和 05_grpo_rl.py 训练时保持一致，效果才明显

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


base_model_name = "Qwen/Qwen3-0.6B"

adapter_path = "./output/qwen3-grpo"


tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

# 前 4 道来自训练集，最后 2 道是训练集外的新题（看泛化能力）
test_questions = [
    "计算：15 + 27 = ?",
    "计算：8 * 7 = ?",
    "计算：100 - 34 = ?",
    "计算：45 / 5 = ?",
    "计算：7 * 6 = ?",
    "计算：1000 - 999 = ?",
]

for question in test_questions:
    messages = [
        {
            "role": "system",
            "content": "你是数学助手。请只输出'答案：数字'的格式，不要输出其他内容。",
        },
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print(f"问题: {question}")
    print(f"回答: {answer.strip()}")
    print("-" * 40)
