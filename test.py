from numpy.ma.core import masked

from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch
from corpus_reader import load_dataset

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
    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length).to(device)

    model.load_state_dict(torch.load('./model_256_100_percent.pth'))

    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    valid_dataset = load_dataset("stas/openwebtext-10k", split="train", tokenizer=tokenizer, max_length=max_length)
    valid_dataset = valid_dataset.shuffle(seed=39)
    valid_dataset = valid_dataset.select(range(10))

    inputs = valid_dataset["input_ids"].to(device)
    labels = valid_dataset["labels"].to(device)
    loss = torch.nn.CrossEntropyLoss()

    # Generate causal mask (causal attention mask) as a 2D matrix
    causal_mask = model.generate_square_subsequent_mask(max_length).to(device)

    for i in range(10):
        print("text", i)
        print("Context: ", inputs[i])
        print(repr(tokenizer.decode(inputs[i])))
        print("Generated text: ")
        output = model(inputs[i].unsqueeze(0).to(device), mask=causal_mask).squeeze(0)
        print("output", torch.argmax(output, dim=1))
        print(repr(tokenizer.decode(torch.argmax(output, dim=1))))
        print("labels", labels[i][1:])

        loss1 = loss(output[:-1], labels[i][1:])
        print("loss", loss1)
        print("\n")

    # text = "BERLIN — When Sarah Kermer proposed to her girlfriend in March, she knew she was in love, but she did not know when, if ever, Germany would allow them to marry.\nThe answer came early Friday morning, when the lower house of the German"
    # input = tokenizer(text, return_tensors='pt')["input_ids"].to(device)
    # print("input", input)
    # causal_mask = model.generate_square_subsequent_mask(len(input[0])).to(device)
    # output = model(input, mask=causal_mask).squeeze(0)
    # print(torch.argmax(output, dim=1))
    # print(tokenizer.decode(torch.argmax(output, dim=1)))
