from model import MyGPT2, MyGPT, train
from corpus_reader import load_dataset_bookcorpus
from transformers import GPT2Tokenizer
from datasets import load_from_disk
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
    max_length = 512

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    model = MyGPT(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length).to(
        device)
    model = torch.compile(model)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    try:
        dataset = load_from_disk("./bookcorpus_10000_512tokens")
    except FileNotFoundError:
        dataset = load_dataset_bookcorpus("bookcorpus/bookcorpus", split="train", tokenizer=tokenizer, max_length=max_length)

    dataset = dataset.shuffle(seed=32)

    valid_dataset = dataset.select(range(1))

    train_dataset = dataset.select(range(1, len(dataset)))

    train(model, train_dataset, valid_dataset=valid_dataset, num_epochs=1, batch_size=64, learning_rate=6e-4,
          device=device, max_length=max_length, warmup_ratio=0.03)
