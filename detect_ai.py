from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return model, tokenizer

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
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
    return ppl.item()

def analyze_text(text):
    model, tokenizer = load_model_and_tokenizer()
    perplexity = calculate_perplexity(text, model, tokenizer)
    return perplexity

def detect_ai_text():
    print("Enter text to analyze (type 'quit' to exit):")
    while True:
        user_input = input("\nText: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        perplexity = analyze_text(user_input)
        
        if perplexity < 20:
            result = "very likely AI-generated"
            confidence = "high"
        elif perplexity < 25:
            result = "likely AI-generated"
            confidence = "moderate"
        elif perplexity < 30:
            result = "possibly AI-generated"
            confidence = "low"
        elif perplexity < 35:
            result = "not likely AI generated"
            confidence = "moderate"
        elif perplexity > 40:
            result = "not likely AI generated"
            confidence = "high"
        else:
            result = "likely human-written"
            confidence = "moderate"
        
        print(f"Analysis result: {result}")
        print(f"Confidence: {confidence}")
        print(f"Perplexity: {perplexity:.2f}")
        print("Interpretation guide:")
        print("- Perplexity < 25: Very likely AI-generated")
        print("- Perplexity 25-30: Likely AI-generated")
        print("- Perplexity 30-35: Not likely AI generate")
        print("- Perplexity > 40: Likely human-written")
        print("Note: These ranges are approximate and may require adjustment based on specific use cases.")

if __name__ == "__main__":
    print("Detect AI-generated text")
    detect_ai_text()

