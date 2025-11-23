import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
basemodel = "microsoft/Phi-3-mini-4k-instruct"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="eager",   # ← flash-attn 이슈 회피
)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

from peft import PeftModel
realmodel = PeftModel.from_pretrained(base, "/home/chldlsrb08/WAGYU_AI/phi3/checkpoint-150")  # adapter_model.safetensors 로드

realmodel.save_pretrained("./phi3/final")
tok.save_pretrained("./phi3/final")

merged = realmodel.merge_and_unload()  # 💡 LoRA 합쳐서 완제품으로
merged.save_pretrained("./final_model", safe_serialization=True)