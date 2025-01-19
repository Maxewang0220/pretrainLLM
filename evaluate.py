from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from torch.nn.functional import softmax

# perplexity for language model
import torch
import torch.nn.functional as F

from Q_A import qa_data
import random


# perplexity; 1st part of the evaluation
def calculate_perplexity(model, inputs, device='cuda'):
    model.to(device)

    input_ids = inputs.to(device)

    # get logits
    with torch.no_grad():
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

    # softmax
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # get the probability of the target word
    target_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
    # to avoid log(0) issue, add a small epsilon
    if torch.any(target_probs == 0):
        print("Warning: target_probs contains 0 values")

    # 避免 log(0) 的问题，添加一个小的 epsilon
    # epsilon = 1e-10
    epsilon = 0
    log_probs = torch.log(target_probs + epsilon)  # (batch_size, seq_len)

    print(f"log_probs: {log_probs}")

    # 计算每个样本的平均 log 概率
    average_log_prob = log_probs.mean(dim=-1)  # (batch_size)

    # 计算 perplexity
    perplexity = torch.exp(-average_log_prob)  # (batch_size)
    print(f"Perplexity: {perplexity}")

    # 返回平均 perplexity
    return perplexity.mean().item()


def Q_A_probability(model, tokenizer, inputs, device='cuda', qa_data=qa_data):
    model.to(device)
    # get a random question and answer pair
    random_qa = random.choice(qa_data)

    question = random_qa["question"]
    real_answer = random_qa["answer"]

    generated_sequence = predict(model, question, tokenizer, max_length=50, eos_token_id=tokenizer.eos_token_id,
                                 device=device)

    # tokenize the question and answer
    real_answer_tokens = tokenizer(real_answer, truncation=True, max_length=200, return_tensors="pt")
    logits = model(real_answer["input_ids"].to(device))


def get_next_token_distributions(model, input_sentence, tokenizer, max_tokens=10, device='cuda'):
    model.to(device)
    model.eval()

    # Tokenize input sentence and move to device
    input_sequence = tokenizer(input_sentence, return_tensors="pt")["input_ids"].to(device)
    generated_sequence = input_sequence.clone()

    token_distributions = []  # Store probability distributions for each generated token

    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass to get logits
            outputs = model(generated_sequence)
            logits = outputs[:, -1, :]  # Get the logits for the last token in the sequence

            # Convert logits to probabilities
            probs = softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)

            # Store the probability distribution
            token_distributions.append(probs[0].cpu().numpy())

            # Get the next token (argmax or sampling, here using argmax)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append the predicted token to the sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

    return token_distributions


# model.to(device)
# https://huggingface.co/thanhnew2001/everything

# 3nd part of the evaluation
def generate_write_n_sentences(model, tokenizer, device='cuda', num_sentence=10):
    model.to(device)

    rate_prompt = """You are a language expert tasked with evaluating a set of generated sentences. Please rate each sentence based on the following criteria:
Grammar and Syntax (1-10): Does the sentence follow proper grammar and syntax rules?
Semantic Clarity (1-10): Is the meaning of the sentence clear and easy to understand?
Contextual Relevance (1-10): Is the sentence relevant to the given topic or theme?
Creativity and Style (1-10): Does the sentence demonstrate creativity or an appropriate style?
For each sentence, provide a detailed score (1-10) for each category, along with a brief explanation for your ratings.

Here are the sentences to evaluate:

[Sentence 1]
[Sentence 2]
[Sentence 3]
Please respond in the following format:

Sentence 1:

Grammar and Syntax: X/10
Semantic Clarity: X/10
Contextual Relevance: X/10
Creativity and Style: X/10
Reasoning: [Provide a detailed explanation]"""

    # load the CNN/DailyMail dataset as prompt
    cnn_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)

    # write to a file
    with open("generated_sentences.txt", "w") as f:
        f.write(rate_prompt)
        f.write("Generated sentences:\n")
        for i in range(num_sentence):
            sentence_idx = "This is sentence " + str(i) + ":\n"
            # original text
            input_text = cnn_dataset[i]["article"]
            print("Context: ", input_text, '\n')
            # tokenize and truncate to max_length tokens
            encoded = tokenizer(input_text, truncation=True, max_length=200, return_tensors="pt")

            # decode the tokenized input text
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            print("input_text: ", input_text, '\n')

            # string
            print("Generated text: ")
            generated_text = predict(model, input_text, tokenizer, max_length=50, eos_token_id=tokenizer.eos_token_id,
                                     device='cuda')
            print("\n")
            # char lists to string sentence
            generated_text = ''.join(generated_text).strip()

            f.write(sentence_idx)
            f.write(generated_text + "\n")


if __name__ == "__main__":
    # hyper parameters
    vocab_size = 50257
    # size of the embedding (original)
    embedding_size = 768
    # number of transformer blocks (original)
    num_layers = 12
    # number of attention heads (original)
    num_heads = 12
    forward_expansion = 4
    dropout = 0.1
    max_length = 256
    device = 'cuda'

    # instantiate the model
    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)

    model.load_state_dict(torch.load('model_256_100_percent.pth'))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    while True:
        # input text
        input_text = input("Context: ")
        if input_text == 'exit':
            break

        print("Generated text: ")

        # inference
        generated_sequence = predict(model, input_sequence=input_text, tokenizer=tokenizer, max_length=50,
                                     eos_token_id=tokenizer.eos_token_id,
                                     device=device)
