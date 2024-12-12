from model import MyGPT2, train
from corpus_reader import load_dataset, tokenize_corpus, load_dataset_wiki
from transformers import GPT2Tokenizer
import torch

# Press the green button in the gutter to run the script.
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

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length).to(
        device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    train_dataset = load_dataset("Skylion007/openwebtext", split="train[:10%]", tokenizer=tokenizer, max_length=max_length)

    valid_dataset = load_dataset("stas/openwebtext-10k", split="train", tokenizer= tokenizer, max_length=max_length)
    valid_dataset = valid_dataset.shuffle(seed=7)
    valid_dataset = valid_dataset.select(range(10))

    train(model, train_dataset, valid_dataset= valid_dataset, num_epochs=1, batch_size=24, learning_rate=1e-6, device=device, max_length=max_length)
