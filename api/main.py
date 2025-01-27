from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

from fastapi import FastAPI

app = FastAPI()
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


@app.post("/next-token")
def get_next_token(sentence: str, k: int = 10):
    top_k_tokens, top_k_probs = get_k_next_tokens(sentence, model, tokenizer, k=k)

    ret_value = {}
    for i in range(10):
        token = top_k_tokens[i]
        prob = top_k_probs[i]
        ret_value[token] = prob.item()

    return ret_value
