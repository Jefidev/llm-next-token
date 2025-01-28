from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from pydantic import BaseModel
from fastapi import FastAPI
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloomz-560M", return_dict_in_generate=True, output_attentions=True
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560M")


def get_k_next_tokens(input_text, model, tokenizer, k=10):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    last_token_logits = logits[:, -1, :]
    top_k_probs, top_k_indices = torch.topk(
        torch.softmax(last_token_logits, dim=-1), k + 1
    )

    top_k_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices[0]]
    top_k_probs = top_k_probs[0].numpy()

    # Remove the end-of-text token if it exists
    for i, token in enumerate(top_k_tokens):
        if token == "</s>":
            top_k_tokens.pop(i)
            top_k_probs = np.delete(top_k_probs, i)
            break

    if len(top_k_tokens) > k:
        top_k_tokens = top_k_tokens[:k]
        top_k_probs = top_k_probs[:k]
    print(top_k_tokens[0])
    return top_k_tokens, top_k_probs


@app.get("/")
def read_root():
    return {"Hello": "World"}


class NextTokenRequest(BaseModel):
    sentence: str
    k: int = 10


class AttentionRequest(BaseModel):
    sentence: str


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


@app.post("/attention-score")
def get_attention_score(request: AttentionRequest):

    sentence = request.sentence
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True)

    # Forward pass to get outputs with attentions
    outputs = model(**inputs)
    attentions = outputs.attentions  # List of attention weights for each layer

    # Select a fixed layer and two random heads
    fixed_layer = -1  # Last layer
    num_heads = attentions[fixed_layer].shape[1]
    choose_head = 5
    head_indices = random.sample(range(num_heads), choose_head)  # Pick n random heads
    # Attention scores for the selected heads

    attention_heads = [
        attentions[fixed_layer][0, head_indices[i]] for i in range(choose_head)
    ]  # Shape: [seq_len, seq_len]

    # Aggregate attention scores per token (e.g., mean over columns)
    attention_aggs = [
        attention_heads[i].mean(dim=0) for i in range(choose_head)
    ]  # Shape: [seq_len] Note: other way to aggregate???

    # Normalize attention scores to 0-1 for visualization
    attention_agg_norm = [
        (attention_aggs[i] - attention_aggs[i].min())
        / (attention_aggs[i].max() - attention_aggs[i].min())
        for i in range(choose_head)
    ]

    # Convert tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean_tokens = [
        token.replace("Ġ", "") for token in tokens
    ]  # Remove special tokens for readability

    results = {}
    results["tokens"] = clean_tokens
    for i in range(choose_head):
        results[f"attention_score_head_{i}"] = attention_agg_norm[i].tolist()

    return results
