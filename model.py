import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


if __name__ == "__main__":
    # hyper parameters
    vocab_size = 50257
    embedding_size = 768
    # number of transformer blocks
    num_layers = 12
    # number of attention heads
    num_heads = 12
    forward_expansion = 4
    dropout = 0.1
    max_length = 512

    # instantiate the model
    model = MyGPT2(vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length)


