import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Tokenizer


# define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()

        # multi-head attention
        self.attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)

        # add & norm
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # shape of x: (sequence_length, batch_size, embedding_size)
        # adjust the shape
        x = x.transpose(0, 1)

        # input x to multi-head attention & add mask if required
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)

        # residual connection and dropout
        x = x + self.dropout(attention_output)

        # normalization
        x = self.norm1(x)

        # feed forward
        forward_output = self.feed_forward(x)

        # residual connection and normalization
        out = self.norm2(x + self.dropout(forward_output))

        return out.transpose(0, 1)


# model implementation
class MyGPT2(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length):
        super(MyGPT2, self).__init__()

        # size of the embedding
        self.embedding_size = embedding_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        # stack num_layers Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        # fit embedding size to vocab size
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(x.device)

        # add token and position embeddings
        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        # input to transformer blocks
        for layer in self.layers:
            out = layer(out, mask)

        logits = self.fc_out(out)
        return logits

    def generate_square_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).to(torch.float)
        return mask


# def train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda', max_length=512):
#     model.to(device)
#     model.train()
#
#     seq_length = max_length
#
#     # Generate causal mask (causal attention mask) as a 2D matrix
#     causal_mask = model.generate_square_subsequent_mask(max_length).to(device)  # Shape: [seq_length, seq_length]
#
#     # DataLoader for batching
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     # Loss function
#     criterion = nn.CrossEntropyLoss()
#
#     # GradScaler for mixed precision
#     scaler = GradScaler()
#
#     # Training loop
#     for epoch in range(num_epochs):
#         total_loss = 0
#         t1 = time.time()
#         for batch_idx, batch in enumerate(dataloader):
#             # Extract inputs and targets from batch
#             x = batch['input_ids'].to(device)  # Input token IDs
#             y = batch['labels'].to(device)  # Target labels
#
#             # Forward and backward pass with mixed precision
#             optimizer.zero_grad()
#
#             with autocast():  # Enable mixed precision
#                 outputs = model(x, mask=causal_mask)
#
#                 # Shift logits and labels for causal language modeling
#                 outputs = outputs[:, :-1, :].contiguous()
#                 y = y[:, 1:].contiguous()
#
#                 # Reshape outputs and targets for calculating loss
#                 outputs = outputs.view(-1, outputs.size(-1))
#                 y = y.view(-1)
#
#                 # Compute loss
#                 loss = criterion(outputs, y)
#
#             # Backward pass with gradient scaling
#             scaler.scale(loss).backward()
#
#             # Optimization step
#             scaler.step(optimizer)
#             scaler.update()
#
#             total_loss += loss.item()
#
#         # Print loss per epoch
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
#         print(f"Time taken for epoch: {time.time() - t1:.2f} sec\n")
#
#     # Save the model
#     torch.save(model.state_dict(), './model/model.pth')
def train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda', max_length=512):
    model.to(device)
    model.train()

    seq_length = max_length

    # Generate causal mask (causal attention mask) as a 2D matrix
    causal_mask = model.generate_square_subsequent_mask(max_length).to(device)

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # GradScaler for mixed precision
    scaler = GradScaler()

    # Total number of batches
    total_batches = len(dataloader)

    # 每完成10%保存一次
    save_intervals = [int(total_batches * (i / 10)) for i in range(1, 11)]  # 保存点：[10%, 20%, ..., 100%]
    save_intervals_idx = 0  # 当前进度检查点索引

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        t1 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets from batch
            x = batch['input_ids'].to(device)  # Input token IDs
            y = batch['labels'].to(device)  # Target labels

            # Forward and backward pass with mixed precision
            optimizer.zero_grad()

            with autocast():
                outputs = model(x, mask=causal_mask)

                # Shift logits and labels for causal language modeling
                outputs = outputs[:, :-1, :].contiguous()
                y = y[:, 1:].contiguous()

                # Reshape outputs and targets for calculating loss
                outputs = outputs.view(-1, outputs.size(-1))
                y = y.view(-1)

                # Compute loss
                loss = criterion(outputs, y)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Optimization step
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Check if we need to save the model at this batch
            if save_intervals_idx < len(save_intervals) and (batch_idx + 1) == save_intervals[save_intervals_idx]:
                model_name = f'model_{save_intervals_idx + 1}0_percent.pth'
                save_path = f'/kaggle/working/{model_name}'
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at {save_path} after {save_intervals_idx + 1}0% of training.")

                # 新增：打印平均损失值
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"Training progress: {save_intervals_idx + 1}0%, Average Loss so far: {avg_loss_so_far:.4f}")

                save_intervals_idx += 1  # Move to the next save interval

        # Print loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Time taken for epoch: {time.time() - t1:.2f} sec\n")

    # Save the final model
    final_model_name = 'final_model.pth'
    torch.save(model.state_dict(), f'/kaggle/working/{final_model_name}')
    print(f"Final model saved at /kaggle/working/{final_model_name}")


# Inference function
def predict(model, input_sequence, tokenizer, max_length=50, eos_token_id=None, device='cuda'):
    model.to(device)
    model.eval()

    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode input text to token IDs
    input_sequence = tokenizer(input_sequence, return_tensors="pt")
    generated_sequence = input_sequence["input_ids"].to(device)

    # Generate text
    generated_text = []

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated_sequence)

            # Get the predicted next token (take the last token in the sequence)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

            # Decode the new token
            new_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated_text += new_token
            print(new_token, end="")

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    print("\n")
    return generated_text


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
    max_length = 512
    device = 'cuda'

    # instantiate the model
    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)

    # train
    dataset = None
    train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device=device)

    # # inference
    # model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)
    # model.load_state_dict(torch.load('./model/model.pth'))
    #
    # # input text
    # input_text = "Once upon a time"
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # input_ids = tokenizer.encode(input_text, return_tensors='pt')
    #
    # #generate text
    # output = predict(model, input_ids, max_length=100, device='cuda')
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)
