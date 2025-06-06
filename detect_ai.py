"""
Core module for AI text detection using perplexity.

This module handles the loading of a pre-trained GPT-2 model and tokenizer,
and provides functions to calculate the perplexity of a given text.
Perplexity is used as a measure to distinguish between AI-generated and
human-written text.
"""
import os
import logging
from typing import Optional
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_NAME = 'distilgpt2'  # Use the smaller DistilGPT-2 model from Hugging Face

model: Optional[GPT2LMHeadModel] = None
tokenizer: Optional[GPT2TokenizerFast] = None

def load_model_and_tokenizer() -> None:
    """
    Loads the GPT-2 model and tokenizer from Hugging Face Transformers.

    Uses the `MODEL_NAME` global variable to specify the model.
    The loaded model and tokenizer are stored in global variables `model` and `tokenizer`.
    This function is called automatically when the module is imported.
    It ensures that the model and tokenizer are loaded only once.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        logging.info(f"Loading DistilGPT-2 model '{MODEL_NAME}' from Hugging Face")
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir='./models')
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, cache_dir='./models')
        logging.info("Model loaded successfully")

# Load the model when the module is imported
load_model_and_tokenizer()

def calculate_perplexity(text: str, language_model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> float:
    """
    Calculates the perplexity of a given text using a specified language model and tokenizer.

    Perplexity is a measure of how well a probability model predicts a sample.
    Lower perplexity generally indicates text that is more predictable or fluent,
    often characteristic of AI-generated text from models like GPT.

    The calculation is performed using a sliding window approach to handle texts
    longer than the model's maximum sequence length. The negative log-likelihoods
    from each window are aggregated to compute the overall perplexity.

    Args:
        text: The input string for which perplexity is to be calculated.
        language_model: The pre-trained GPT-2 language model (e.g., GPT2LMHeadModel).
        tokenizer: The tokenizer corresponding to the language model (e.g., GPT2TokenizerFast).

    Returns:
        The calculated perplexity score as a float.
    """
    # This try-except block was removed in a previous step, but the instruction asks to keep logging.
    # If an error occurs, it will propagate up to the caller.
    # logging.debug(f"Analyzing text: {text}") can be added back if needed for debugging.
    # However, since this function is called by analyze_text, which also logs, it might be redundant.
    # For now, I will keep it as is, without the try-except block and without duplicating the logging.
    logging.debug(f"Analyzing text: {text}")
        
        token_encodings = tokenizer(text, return_tensors='pt')
        max_sequence_length = language_model.config.n_positions
        window_stride = 512
        total_sequence_length = token_encodings.input_ids.size(1)
        negative_log_likelihoods = []
        previous_window_end = 0
        
        # Iterate over the text using a sliding window approach
        for window_start in range(0, total_sequence_length, window_stride):
           
            window_end = min(window_start + max_sequence_length, total_sequence_length)
            target_length = window_end - previous_window_end
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

def analyze_text(text: str) -> float:
    """
    High-level interface for analyzing text to calculate its perplexity.

    This function serves as a simple entry point to the perplexity calculation.
    It ensures that the global model and tokenizer are loaded (if not already)
    and then calls `calculate_perplexity` with the provided text.

    Args:
        text: The input string to analyze.

    Returns:
        The perplexity score of the text as a float.

    Raises:
        RuntimeError: If the model and tokenizer cannot be loaded.
    """
    if model is None or tokenizer is None:
        # This case should ideally not happen if load_model_and_tokenizer is always called at startup.
        # However, to satisfy type hinting for calculate_perplexity, we ensure model and tokenizer are not None.
        logging.error("Model or tokenizer not loaded. Attempting to load now.")
        load_model_and_tokenizer()
        # If still None after attempting to load, raise an error or handle appropriately.
        if model is None or tokenizer is None:
            raise RuntimeError("Failed to load model and tokenizer.")

    perplexity = calculate_perplexity(text, model, tokenizer)
    logging.info(f"Perplexity calculated: {perplexity}")
    return perplexity