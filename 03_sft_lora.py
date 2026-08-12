import torch

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


model_name = "Qwen/Qwen3-0.6B"

# ============================
# 1. 加载自己的数据
# ============================

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)

print(dataset[0])


# ============================
# 2. LoRA 配置
# ============================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules="all-linear",

    task_type="CAUSAL_LM"
)


# ============================
# 3. SFT训练参数
# ============================

training_args = SFTConfig(
    output_dir="./output/qwen3-home-lora",

    num_train_epochs=3,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    logging_steps=1,
    save_steps=50,

    assistant_only_loss=True,

    bf16=torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),

    fp16=torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
)


# ============================
# 4. 构造Trainer
# ============================

trainer = SFTTrainer(
    model=model_name,

    args=training_args,

    train_dataset=dataset,

    peft_config=lora_config
)


# ============================
# 5. 开始训练
# ============================

trainer.train()


# ============================
# 6. 保存
# ============================

trainer.save_model(
    "./output/qwen3-home-lora"
)