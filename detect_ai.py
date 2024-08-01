from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return model, tokenizer

#perplexity is a measure of how well a model predicts a sample from itself therby measuring the model's uncertainty. A higher perplexity indicates a higher uncertainty in the model's predictions indicating more likely a human-written text.
def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

#using a sliding window to stream encodingscalculate perplexity
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
# formula for perplexity is the exponentiation of the average negative log-likelihood per token
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def analyze_text(text):
    model, tokenizer = load_model_and_tokenizer()
    perplexity = calculate_perplexity(text, model, tokenizer)
    return perplexity
