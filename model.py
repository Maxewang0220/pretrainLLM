import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


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


# Training function
def train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda'):
    model.to(device)
    model.train()

    num_heads = model.layers[0].attention.num_heads  # Assuming TransformerBlock has `attention.num_heads`

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        i = 1
        t1 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and attention masks from batch
            x = batch['input_ids'].to(device)  # Input token IDs
            y = batch['labels'].to(device)  # Target labels
            attention_mask = batch['attention_mask'].to(device)  # Padding mask

            # Generate causal mask (for time step dependencies)
            seq_length = x.shape[1]
            causal_mask = model.generate_square_subsequent_mask(seq_length).to(device)

            # Combine causal mask and attention mask
            # Convert attention_mask to 3D shape for compatibility with causal mask
            expanded_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length,
                                                                         seq_length)  # Shape: [batch_size, seq_length, seq_length]
            causal_mask = causal_mask.unsqueeze(0).expand(attention_mask.size(0), -1,
                                                          -1)  # Shape: [batch_size, seq_length, seq_length]
            combined_mask = causal_mask * expanded_attention_mask  # Apply both masks
            combined_mask = combined_mask.to(torch.float)
            combined_mask = combined_mask.masked_fill(combined_mask == 0, float('-inf')).masked_fill(combined_mask == 1,
                                                                                                     float(0.0))

            # Add a num_heads dimension and expand it
            combined_mask = combined_mask.unsqueeze(1).expand(-1, num_heads, -1,
                                                              -1)  # Shape: [batch_size, num_heads, seq_length, seq_length]

            # Flatten batch and num_heads dimensions
            combined_mask = combined_mask.reshape(batch_size * num_heads, seq_length,
                                                  seq_length)  # Shape: [batch_size * num_heads, seq_length, seq_length]

            # Forward pass with combined mask
            outputs = model(x, mask=combined_mask)

            # Shift logits and labels for causal language modeling
            outputs = outputs[:, :-1, :].contiguous()
            y = y[:, 1:].contiguous()

            # Reshape outputs and targets for calculating loss
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs
            y = y.view(-1)  # Flatten targets

            # Compute loss
            loss = criterion(outputs, y)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print("Step", i)
            i += 1

        # Print loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Time taken for epoch: {time.time() - t1:.2f} sec\n")

    # Save the model
    torch.save(model.state_dict(), './model/model.pth')


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

    # train
    dataset = None
    train(model, dataset, num_epochs=3, batch_size=32, learning_rate=1e-4, device='cuda')

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
