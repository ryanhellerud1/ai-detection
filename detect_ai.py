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

def calculate_perplexity(text, language_model, tokenizer):
    try:
        logging.debug(f"Analyzing text: {text}")
        
        # Tokenize the input text
        token_encodings = tokenizer(text, return_tensors='pt')
        
        # Get the maximum sequence length the model can handle
        max_sequence_length = language_model.config.n_positions
        
        # Set the stride for sliding window approach
        window_stride = 512
        
        # Get the total length of the input sequence
        total_sequence_length = token_encodings.input_ids.size(1)

        negative_log_likelihoods = []
        previous_window_end = 0
        
        # Iterate over the text using a sliding window approach
        for window_start in range(0, total_sequence_length, window_stride):
            # Determine the end of the current window
            window_end = min(window_start + max_sequence_length, total_sequence_length)
            
            # Calculate the target length for this window
            target_length = window_end - previous_window_end
            
            # Extract the input ids for the current window
            input_ids = token_encodings.input_ids[:, window_start:window_end]
            
            # Create target ids, masking out previously processed tokens
            target_ids = input_ids.clone()
            target_ids[:, :-target_length] = -100

            # Calculate the loss for this window
            with torch.no_grad():
                model_output = language_model(input_ids, labels=target_ids)
                window_negative_log_likelihood = model_output.loss * target_length

            negative_log_likelihoods.append(window_negative_log_likelihood)

            previous_window_end = window_end
            if window_end == total_sequence_length:
                break

        # Calculate the perplexity using the accumulated negative log likelihoods
        perplexity = torch.exp(torch.stack(negative_log_likelihoods).sum() / total_sequence_length)
        
        logging.debug(f"Perplexity score calculated: {perplexity.item()}")
        return perplexity.item()
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