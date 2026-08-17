# 05_grpo_rl.py
# GRPO 强化学习训练脚本（延续 03_sft_lora.py 的风格）
#
# 学习要点：
#   1. SFT 是"模仿"，RL 是让模型自己采样 -> 打分 -> 按分数调整策略
#   2. GRPO：对同一个问题采样 G 个回答，组内互相比较得出"相对优势"
#   3. 奖励函数是 RL 的核心，这里演示两个最简单的规则式奖励
#
# 依赖：trl >= 0.16（提供 GRPOTrainer / GRPOConfig）
# 官方文档：https://huggingface.co/docs/trl/grpo_trainer

import re
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig


model_name = "Qwen/Qwen3-0.6B"

# ============================
# 1. 准备数据：把问题套上 Chat 模板
#    （关闭 thinking，让生成结果干净、奖励容易判定）
# ============================

tokenizer = AutoTokenizer.from_pretrained(model_name)


def build_prompt(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是数学助手。请只输出'答案：数字'的格式，不要输出其他内容。",
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


dataset = load_dataset(
    "json",
    data_files="data/math_train.jsonl",
    split="train",
)


def preprocess(example):
    return {
        # prompt 会作为生成输入；answer 是额外列，会自动以 kwargs 传给奖励函数
        "prompt": build_prompt(example["question"]),
        "answer": example["answer"],
    }


dataset = dataset.map(preprocess)

print("===== 示例 prompt =====")
print(repr(dataset[0]["prompt"]))
print("示例 answer:", dataset[0]["answer"])


# ============================
# 2. 奖励函数（GRPO 的核心）
#    每个函数接收 (prompts, completions, **kwargs)，
#    返回与 completions 等长的奖励列表（越大越好）
# ============================

def format_reward(prompts, completions, **kwargs):
    """格式奖励：回答里包含'答案：'就给 1 分"""
    rewards = []
    for completion in completions:
        rewards.append(1.0 if "答案" in completion else 0.0)
    return rewards


def extract_number(text: str):
    """提取'答案：'后面的第一个数字"""
    match = re.search(r"答案[:：]\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def accuracy_reward(prompts, completions, answers, **kwargs):
    """正确性奖励：最终答案与标准答案一致就给 1 分"""
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_number(completion)
        if predicted is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if abs(predicted - float(answer)) < 1e-6 else 0.0)
    return rewards


# ============================
# 3. LoRA 配置（和 SFT 脚本一致）
# ============================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# ============================
# 4. GRPO 训练参数
# ============================

training_args = GRPOConfig(
    output_dir="./output/qwen3-grpo",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,           # 每个 prompt 采样 8 个回答组成一组（GRPO 的关键参数）
    max_completion_length=128,   # 每个回答最多生成多少 token
    learning_rate=1e-5,          # RL 学习率通常比 SFT 低一个量级
    beta=0.04,                   # KL 惩罚系数：防止模型偏离 SFT 模型太远
    logging_steps=1,
    save_steps=50,
    bf16=torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported(),
)


# ============================
# 5. 构造 Trainer 并训练
# ============================

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[format_reward, accuracy_reward],
    peft_config=lora_config,
)

trainer.train()


# ============================
# 6. 保存 LoRA 适配器
# ============================

trainer.save_model("./output/qwen3-grpo")
print("\n训练完成，适配器已保存到 ./output/qwen3-grpo")
print("用 04_test_lora.py 测试时，把 adapter_path 改成该路径即可")
