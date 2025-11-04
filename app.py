from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL = "/home/chldlsrb08/final_model"
model = AutoModelForCausalLM.from_pretrained(MODEL,
                                             dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

app = FastAPI()

class chatRequest(BaseModel):
    question: str
    max_newtokens = 256

@app.get("/")
def root():
    return "ok"

@app.post("/ask")
async def ask(req: chatRequest):
    try:
        messages = [
            {"role": "user", "content" : req.question}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = 'pt'
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens = req.max_newtokens,
                temperature = 0.7,
                top_p = 0.9,
                do_sample = True
            )
    except Exception as e:
        print(f"error : {e}")