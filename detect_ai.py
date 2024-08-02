import os
import logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
    return model, tokenizer

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
        logging.error(f"Error analyzing text: {e}")
        return None

def analyze_text(text):
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        logging.error("Failed to load model or tokenizer")
        return None
    try:
        perplexity = calculate_perplexity(text, model, tokenizer)
        return perplexity
    except Exception as e:
        logging.error(f"Error analyzing text: {e}")
        return None