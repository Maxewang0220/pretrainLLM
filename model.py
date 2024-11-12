import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader


#define transformer block
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

        #normalization
        x = self.norm1(x)

        # feed forward
        forward_output = self.feed_forward(x)

        # residual connection and normalization
        out = self.norm2(x + self.dropout(forward_output))

        return out.transpose(0, 1)

#model implementation
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
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        # add token and position embeddings
        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        # input to transformer blocks
        for layer in self.layers:
            out = layer(out, mask)

        logits = self.fc_out(out)
        return logits

    def generate_square_subsequent_mask(size):
        mask = torch.tril(torch.ones(size, size)).to(torch.bool)
        return mask


# Training function
def train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda'):
    model.to(device)
    model.train()

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Generate mask for training
            mask = model.generate_square_subsequent_mask(x.shape[1]).to(device)

            # Forward pass
            outputs = model(x, mask=mask)

            # Reshape outputs and targets for calculating loss
            outputs = outputs.view(-1, outputs.size(-1))
            y = y.view(-1)

            # Compute loss
            loss = criterion(outputs, y)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Inference function
def predict(model, input_sequence, max_length=50, device='cuda'):
    model.to(device)
    model.eval()

    generated_sequence = input_sequence.to(device)
    mask = model.generate_square_subsequent_mask(generated_sequence.shape[1]).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated_sequence, mask=mask)

            # Get the predicted next token (take the last token in the sequence)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

            # Update mask
            mask = model.generate_square_subsequent_mask(generated_sequence.shape[1]).to(device)

    return generated_sequence.cpu()

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

    # instantiate the model
    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)


