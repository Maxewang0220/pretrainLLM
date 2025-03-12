from transformers import GPT2Tokenizer
from model_refer import GPT2
import torch

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
    # model.load_state_dict(torch.load("GPT_Alpaca_512_100_percent.pth"))
    model.load_state_dict(torch.load("GPT_512_50_2_percent.pth"))
    model.eval()

    while True:
        input_sentence = input("Context: ")

        token_id = tokenizer.encode(input_sentence)
        input_data = torch.reshape(torch.tensor(token_id, device=device), [1, -1])
        predicted = model.generate(input_data, generate_len, 1.0)
        print("Generated text:\n-------------------")
        print(tokenizer.decode(predicted.cpu().numpy()[0]))
