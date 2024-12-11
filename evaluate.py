from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from model import MyGPT2, predict
#  文件名 有evaluate，库 也有evaluate，要改名之类的处理冲突================================
from evaluate import load
import torch.nn.functional as F
from transformers import GPT2Tokenizer


def rogue_score(model, tokenizer):
    # load the CNN/DailyMail dataset
    cnn_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
    # load the ROUGE metric
    rouge = load("rouge", trust_remote_code=True)
    # get the length of the dataset
    len_dataset = len(cnn_dataset)

    for i in range(1):
        # original text
        input_text = cnn_dataset[i]["article"]
        # tokenize and truncate to max_length tokens
        encoded = tokenizer(input_text, truncation=True, max_length=200, return_tensors="pt")

        # decode the tokenized input text
        input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

        # human summary
        human_summary = cnn_dataset[i]["highlights"]

        # string
        generated_summary = predict(model, input_text, tokenizer, max_length=50, eos_token_id=tokenizer.eos_token_id,
                                    device='cuda')
        # char lists to string sentence
        generated_summary = ''.join(generated_summary).strip()

        rouge_score = rouge.compute(
            predictions=[generated_summary],  # list of predictions
            references=[human_summary]  # list of references
        )
        print("ROUGE Score: \n", rouge_score)
        # print("input_text: {}\n".format(input_text))
        # print("human_summary: {}\n".format(human_summary))
        # print("generated_summary: {}\n".format(generated_summary))

    # average ROUGE score
    average_rouge = sum(rouge_score.values()) / len(rouge_score)
    print(f"Average ROUGE score: {average_rouge}")


# ttr for language richness
def calculate_ttr(word_list):
    unique_words = set(word_list)
    ttr = len(unique_words) / len(word_list) if len(word_list) > 0 else 0
    return ttr


def generate_text_and_calculate_ttr(model, tokenizer, num_samples=10, max_length=100, device='cuda'):
    # param num_samples: Number of samples to generate
    model.to(device)

    num_samples = 1

    # store the TTR scores for each generated text
    ttr_scores = []

    # because of greedy choice in generation, sentences generated stay same no matter how many times run
    # generate text and calculate TTR for each sample
    for i in range(num_samples):
        prompt = "We are a fast-growing B2B market research Agency working with brands to generate customized marketing campaigns. We "

        # model generates texts
        generated_text = predict(model, prompt, tokenizer, max_length=max_length, eos_token_id=tokenizer.eos_token_id,
                                 device='cuda')

        # char list to word list
        generated_text = ''.join(generated_text).split(' ')

        # ttr for each generated text
        ttr = calculate_ttr(generated_text)
        ttr_scores.append(ttr)

        print(f"Sample {i + 1}:")
        print(f"TTR: {ttr}")
        print("-" * 50)

    # average TTR
    average_ttr = sum(ttr_scores) / len(ttr_scores) if ttr_scores else 0
    print(f"Average TTR across {num_samples} samples: {average_ttr}")


# perplexity for language model
def calculate_perplexity(model, tokenizer, device='cuda'):
    model.to(device)
    input = "I like apples on the tree"
    encoded = tokenizer(input, return_tensors='pt')

    input_ids = encoded["input_ids"].to(device)

    logits = model(input_ids)

    # softmax
    probs = F.softmax(logits, dim=-1)

    # select max probability
    max_prob = torch.max(probs, dim=-1)

    # 提取 max_prob 中的 indices
    max_indices = max_prob.indices.squeeze(0)  # 移除 batch 维度
    print(f"max_indices: {max_indices}")

    # decode max_indices
    max_tokens = [tokenizer.decode([idx]) for idx in max_indices.tolist()]
    print(f"max_tokens: {max_tokens}")

    # to get the probability of the target token
    target_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    print(f"target_probs: {target_probs}")

    # log_probs = target_probs.log()  # [B, T]
    log_probs = (target_probs + 1e-9).log()

    # average log probability
    average_log_prob = log_probs.mean(dim=-1)  # [B]
    print(f"average_log_prob: {average_log_prob}")

    # perplexity
    perplexity = torch.exp(-average_log_prob)  # [B]
    print(f"Perplexity: {perplexity.item()}")


if __name__ == '__main__':
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

    model.load_state_dict(torch.load('model_1_30_percent.pth'))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # =======================evaluation part=====================================#
    # calculate ROGUE score
    rogue_score(model, tokenizer)

    # calculate TTR
    generate_text_and_calculate_ttr(model, tokenizer, num_samples=10, max_length=100, device='cuda')

    # calculate perplexity with cross entropy loss
    calculate_perplexity(model, tokenizer, device='cuda')
    # ==============================================================#

    # generation test
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
