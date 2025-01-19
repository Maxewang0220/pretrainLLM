from itertools import takewhile

from numpy.ma.core import masked

from model import MyGPT2, predict
from transformers import GPT2Tokenizer
import torch
from corpus_reader import load_dataset
from evaluate import calculate_perplexity, generate_write_n_sentences
from Q_A import qa_data
from evaluate import get_next_token_distributions
import numpy as np

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

    model.load_state_dict(torch.load('model_256_100_percent.pth'))

    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    valid_dataset = load_dataset("stas/openwebtext-10k", split="train", tokenizer=tokenizer, max_length=max_length)
    valid_dataset = valid_dataset.shuffle(seed=8)
    valid_dataset = valid_dataset.select(range(10))

    inputs = valid_dataset["input_ids"].to(device)
    labels = valid_dataset["labels"].to(device)
    loss = torch.nn.CrossEntropyLoss()

    # calcullate deals with one, so to process with 10 needed here or in the function itself

    # 1st stcalculate perplexity with cross entropy loss
    # mean_perplexity = calculate_perplexity(model, inputs, device)
    # print("mean perplexity", mean_perplexity)

    # 2nd calculate Q_A token probability
    # Q_A_probability(model, tokenizer, inputs, device='cuda', qa_data=qa_data)
    # input_sentence = "The capital of France is"

    get_next_token_distributions(model, tokenizer, max_tokens=10, device=device)

    # # 3rd generate and write n sentences
    # generate_write_n_sentences(model, tokenizer, device, num_sentence=10)

    # # Generate causal mask (causal attention mask) as a 2D matrix
    # causal_mask = model.generate_square_subsequent_mask(max_length).to(device)
    #
    # # for i in range(10):
    # #     if i == 9:
    # #         print("text", i)
    # #         print("Context: ", inputs[i])
    # #         print(tokenizer.decode(inputs[i]))
    # #         print("Generated text: ")
    # #         output = model(inputs[i].unsqueeze(0).to(device), mask=causal_mask).squeeze(0)
    # #         print("output", torch.argmax(output, dim=1))
    # #         print(tokenizer.decode(torch.argmax(output, dim=1)))
    # #         # print("labels", labels[i][1:])
    # #
    # #         loss1 = loss(output[:-1], labels[i][1:])
    # #         print("loss", loss1)
    # #         print("\n")
    #
    # text = "The process starts with Alice"
    # input = tokenizer(text, return_tensors='pt')["input_ids"].to(device)
    # print("input", input)
    # causal_mask = model.generate_square_subsequent_mask(len(input[0])).to(device)
    # output = model(input, mask=causal_mask).squeeze(0)
    # print(torch.argmax(output, dim=1))
    # print(tokenizer.decode(torch.argmax(output, dim=1)))
    #
    # print(causal_mask)
