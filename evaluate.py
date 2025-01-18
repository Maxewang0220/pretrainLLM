from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# perplexity for language model
import torch
import torch.nn.functional as F


def calculate_perplexity(model, tokenizer, inputs, device='cuda'):
    model.to(device)

    input_ids = inputs.to(device)

    # get logits
    with torch.no_grad():
        logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)

    # softmax
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # get the probability of the target word
    target_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

    # 避免 log(0) 的问题，添加一个小的 epsilon
    epsilon = 1e-9
    log_probs = torch.log(target_probs + epsilon)  # (batch_size, seq_len)

    # 计算每个样本的平均 log 概率
    average_log_prob = log_probs.mean(dim=-1)  # (batch_size)

    # 计算 perplexity
    perplexity = torch.exp(-average_log_prob)  # (batch_size)

    # 返回平均 perplexity
    return perplexity.mean().item()

    # 示例调用
    # tokenizer = ...  # 你的 tokenizer
    # model = ...      # 你的模型
    # inputs = tokenizer("示例文本", return_tensors="pt")["input_ids"]
    # perplexity = calculate_perplexity(model, tokenizer, inputs)
    # print(f"Perplexity: {perplexity}")

    # def Q_A_probability(model, tokenizer, inputs, device='cuda')
    model.to(device)


# def generate_n_sentences(model, tokenizer, device, sentence_num=10):


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
