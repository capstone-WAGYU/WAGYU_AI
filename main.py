## ※ !! Deprecated !! 서버는 aiback 리포지토리 사용

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = BASE_DIR / "augment" / "통합_데이터셋_증강.json"
OUTPUT_DIR = BASE_DIR / "qwen2.5-7b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, peft_config)
model.train()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    max_grad_norm=0.3,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=str(OUTPUT_DIR),
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=2000,
    save_steps=50,
    save_strategy="steps",
    report_to="tensorboard",
)

dataset = load_dataset("json", data_files=str(DATASET_PATH))["train"]

def format_qa(example):
    q = example.get("question", "")
    a = example.get("answer", "")
    messages = [
        {"role": "system", "content": "너는 WAGYU 서비스를 보조하는 AI 어시스턴트다."},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_qa)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer = tokenizer,
    max_seq_length=2048,
    args=training_args,
    formatting_func=lambda ex: ex["text"],
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
