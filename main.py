from model import MyGPT2, train
from corpus_reader import load_dataset, tokenize_corpus
from transformers import GPT2Tokenizer

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

    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)

    dataset = load_dataset("upstage/Pretraining_Dataset", split="train[:10%]")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the dataset
    tokenized_dataset = tokenize_corpus(dataset, tokenizer, max_length=max_length)

    train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda')
