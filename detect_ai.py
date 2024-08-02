import os
import logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gpt2_model')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'gpt2_tokenizer')

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        logging.info(f"Loading GPT-2 model from {MODEL_PATH}")
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)
        logging.info("Model loaded successfully")

# Load the model when the module is imported
load_model_and_tokenizer()

def calculate_perplexity(text, model, tokenizer):
    try:
        logging.debug(f"Analyzing text: {text}")
        encodings = tokenizer(text, return_tensors='pt')
        max_length = model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        logging.debug(f"Perplexity score calculated: {ppl.item()}")
        return ppl.item()
    except Exception as e:
        logging.error(f"Error calculating perplexity: {str(e)}", exc_info=True)
        return None

def analyze_text(text):
    try:
        perplexity = calculate_perplexity(text, model, tokenizer)
        logging.info(f"Perplexity calculated: {perplexity}")
        return perplexity
    except Exception as e:
        logging.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return None