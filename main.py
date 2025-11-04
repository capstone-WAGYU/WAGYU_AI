import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

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
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "right"

output_dir = "./WAGYU_AI/phi3"
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
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=2000,
    save_steps=50,
    save_strategy="steps",
    report_to="tensorboard",
)

dataset = load_dataset("json", data_files="/home/chldlsrb08/WAGYU_AI/augment/통합_데이터셋_증강.json")["train"]
def format_qa(example):
    q = example.get("question", "")
    a = example.get("answer", "")
    return {"text": f"질문: {q}\n답변: {a}"}

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
tokenizer.save_pretrained(output_dir)
