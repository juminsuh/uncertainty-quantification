# This script exists just to load models faster
import functools
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

@functools.lru_cache()
def load_pretrained_model(model_name, torch_dtype=torch.float16): 
    if model_name == 'llama-7b-hf':
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", device_map = 'auto', cache_dir=None, torch_dtype=torch_dtype) # 이미 load가 된 llama를 불러온다. 
    if model_name == 'llama-13b-hf':
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b", cache_dir=None, torch_dtype=torch_dtype) # 이미 load가 된 llama를 불러온다. 
    return model

@functools.lru_cache()
def load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b", cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"{model_name} tokenizer is not supported.")
    return tokenizer