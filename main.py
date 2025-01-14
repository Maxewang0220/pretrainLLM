from model import MyGPT2,MyGPT,train
from corpus_reader import load_dataset, tokenize_corpus, load_dataset_bookcorpus
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

    model = MyGPT(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length).to(
        device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    dataset = load_dataset_bookcorpus("bookcorpus/bookcorpus", split="train", tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.shuffle(seed=32)

    valid_dataset = dataset.select(range(10))

    train_dataset = dataset.select(range(10, len(dataset)))

    train(model, train_dataset, valid_dataset= valid_dataset, num_epochs=1, batch_size=24, learning_rate=1.5e-4, device=device, max_length=max_length, warmup_ratio=0.03)
