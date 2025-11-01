import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, SFTTrainer
from peft import prepare_model_for_kbit_training

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
    trust_remote_code=False,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = 'right'

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

output_dir = "./phi3"
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir = output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=8000,
    report_to='tensorboard'
)

from trl import SFTTrainer
trainer = SFTTrainer(
    model,
    train_dataset = dataset,
    dataset_text_field = "y 뭔지 적어주기",
    tokenizer = tokenizer,
    max_seq_length = 2048,
    args = training_args
)

trainer.train()
trainer.save_model()

import os
outputdir = os.path.join(output_dir, "runs")
trainer.model.save_pretrained(outputdir)
tokenizer.save_pretrained(outputdir)