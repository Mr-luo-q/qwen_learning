import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


base_model_name = "Qwen/Qwen3-0.6B"

adapter_path = "./output/qwen3-home-lora"


tokenizer = AutoTokenizer.from_pretrained(
    base_model_name
)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)


model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)


messages = [
    {
        "role": "user",
        "content": "我要睡觉了"
    }
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True
).to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=100
)


new_tokens = outputs[0][
    inputs["input_ids"].shape[-1]:
]


print(
    tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )
)