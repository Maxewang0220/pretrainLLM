from model import MyGPT2, predict

from Q_A import qa_data
from model_refer import GPT2

# 导入 Transformers 库的 GPT2 Tokenizer
from transformers import GPT2Tokenizer

# 导入 PyTorch 相关库
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

# 导入 Hugging Face Datasets 库
from datasets import load_dataset

# 其他工具库
import random
import numpy as np


# perplexity; 1st part of the evaluation
def calculate_perplexity(model, inputs, device='cuda'):
    model.to(device)
    model.eval()

    input_ids = inputs.to(device)

    # get logits
    with torch.no_grad():
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

    # softmax
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # get the probability of the target word
    target_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
    if torch.any(target_probs == 0):
        print("Warning: target_probs contains 0 values")

    # 避免 log(0) 的问题，可以考虑添加一个小的 epsilon
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


def get_QA_token_prob(model, tokenizer, max_tokens=10, device='cuda', qa_data=qa_data):
    model.to(device)
    model.eval()

    # get a random question and answer pair
    random_qa = random.choice(qa_data)
    question = random_qa["question"]
    real_answer = random_qa["answer"]
    print("question is : ", question)
    print("real_answer is : ", real_answer)

    answer_tokens = tokenizer(real_answer, truncation=True, max_length=200, return_tensors="pt")["input_ids"].to(
        device)
    print("answer_tokens: ", answer_tokens[0])
    print("shape of answer_tokens: ", answer_tokens.shape)

    # 假设 answer_tokens 是通过 tokenizer 得到的 input_ids
    answer_tokens_ids = answer_tokens[0].tolist()  # 转换为列表
    tokens = tokenizer.convert_ids_to_tokens(answer_tokens_ids)  # 转换为 token
    print("Tokens: ", tokens)

    # Tokenize input sentence and move to device
    input_sequence = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
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

    # convert to numpy array
    # convert to numpy array
    token_distributions_array = np.array(token_distributions)
    print(token_distributions_array.shape)  # 输出 (10, 50257)

    # Convert answer_tokens to CPU and NumPy array
    answer_tokens = answer_tokens.cpu().numpy()  # 转为 NumPy 数组
    print("answer_tokens: ", answer_tokens)  # 输出 answer_tokens:  [464]
    print("len(answer_tokens): ", len(answer_tokens))  # 输出 len(answer_tokens):  1

    # get the probability of the target token
    selected_probs = token_distributions_array[:, answer_tokens]  # Shape: (max_tokens, len(answer_tokens))
    print("Selected probabilities:\n", selected_probs)
    print("Shape: ", selected_probs.shape)

    return token_distributions


# model.to(device)
# https://huggingface.co/thanhnew2001/everything

# 3nd part of the evaluation
def generate_write_n_sentences(model, tokenizer, device='cuda', num_sentence=10):
    model.to(device)
    model.eval()

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
            # max_length=200, generate a lot ( and .
            # encoded = tokenizer(input_text, truncation=True, max_length=200, return_tensors="pt")
            encoded = tokenizer(input_text, truncation=True, max_length=20, return_tensors="pt")

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


if __name__ == '__main__':
    no_mixed = False
    batch_size = 16
    vocab_size = 50257
    max_length = 512
    num_layers = 12
    num_heads = 12
    embedding_size = 768
    forward_expansion = 3072
    embedding_dropout = 0.1
    attention_dropout = 0.1
    residual_dropout = 0.1
    feedforward_dropout = 0.1
    weight_decay = 0.01
    warm_up = 0.03
    generate_len = 128

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2(
        vocab_size=vocab_size,
        d_model=embedding_size,
        block_size=max_length,
        embed_pdrop=embedding_dropout,
        num_heads=num_heads,
        dff=forward_expansion,
        attn_pdrop=attention_dropout,
        resid_pdrop=residual_dropout,
        dropout=feedforward_dropout,
        num_layer=num_layers)

    model.to(device)
    model.load_state_dict(torch.load("GPT_Alpaca_512_100_percent.pth"))
    model.eval()  # ============================================

    valid_dataset = load_dataset("stas/openwebtext-10k", split="train", tokenizer=tokenizer, max_length=max_length)
    valid_dataset = valid_dataset.shuffle(seed=8)
    valid_dataset = valid_dataset.select(range(10))

    inputs = valid_dataset["input_ids"].to(device)
    labels = valid_dataset["labels"].to(device)
    loss = torch.nn.CrossEntropyLoss()

    # calcullate deals with one, so to process with 10 needed here or in the function itself

    # 1st calculate perplexity with cross entropy loss
    mean_perplexity = calculate_perplexity(model, inputs, device)
    print("mean perplexity", mean_perplexity)

    get_QA_token_prob(model, tokenizer, max_tokens=10, device=device)

    # # 3rd generate and write n sentences
    generate_write_n_sentences(model, tokenizer, device, num_sentence=10)

    # while True:
    #     # input text
    #     input_text = input("Context: ")
    #     if input_text == 'exit':
    #         break
    #
    #     print("Generated text: ")
    #
    #     # inference
    #     generated_sequence = predict(model, input_sequence=input_text, tokenizer=tokenizer, max_length=50,
    #                                  eos_token_id=tokenizer.eos_token_id,
    #                                  device=device)
