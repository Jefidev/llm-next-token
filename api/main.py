from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")


def get_k_next_tokens(input_text, model, tokenizer, k=10):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    last_token_logits = logits[:, -1, :]
    top_k_probs, top_k_indices = torch.topk(torch.softmax(last_token_logits, dim=-1), k)

    top_k_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices[0]]
    top_k_probs = top_k_probs[0].numpy()
    return top_k_tokens, top_k_probs


@app.get("/")
def read_root():
    return {"Hello": "World"}


class NextTokenRequest(BaseModel):
    sentence: str
    k: int = 10


@app.post("/next-token")
def get_next_token(request: NextTokenRequest):
    sentence = request.sentence
    k = request.k
    top_k_tokens, top_k_probs = get_k_next_tokens(sentence, model, tokenizer, k=k)

    ret_value = {}
    for i in range(k):
        token = top_k_tokens[i]
        prob = top_k_probs[i]
        ret_value[token] = prob.item()

    return ret_value
