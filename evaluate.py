from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import GPT2Tokenizer


# perplexity for language model
def calculate_perplexity(model, tokenizer, inputs, device='cuda'):
    model.to(device)

    input_ids = inputs

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
    # log_probs = (target_probs + 1e-9).log()  # prevent log(0)
    log_probs = (target_probs).log()

    # average log probability
    average_log_prob = log_probs.mean(dim=-1)  # [B]
    print(f"average_log_prob: {average_log_prob}")

    # perplexity
    perplexity = torch.exp(-average_log_prob)  # [B]
    print(f"Perplexity: {perplexity.item()}")

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
