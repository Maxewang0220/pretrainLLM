import time
import logging
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Tokenizer
from transformers import get_scheduler


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
            nn.GELU(),
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
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask


class GPT2Block(nn.Module):
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout):
        super(GPT2Block, self).__init__()

        # multi-head attention
        self.attn = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout, batch_first=True)
        # GPT-2风格：在进入子层之前做LayerNorm
        self.ln_1 = nn.LayerNorm(embedding_size)
        self.ln_2 = nn.LayerNorm(embedding_size)

        # feed forward: MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: shape (batch_size, seq_length, embedding_size)

        # 1) Pre-LN, Self-Attention
        x = self.ln_1(x)  # GPT-2 style: apply LN before attention
        # batch_first=True，所以不需要转置
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = x + self.drop(attn_out)  # 残差连接

        # 2) Pre-LN, Feed Forward
        m = self.ln_2(x)
        mlp_out = self.mlp(m)
        x = x + self.drop(mlp_out)

        return x


class MyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, max_length=1024):
        super(MyGPT, self).__init__()

        # token & position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        # transformer blocks
        self.layers = nn.ModuleList([
            GPT2Block(embedding_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        # last layer norm (ln_f)
        self.ln_f = nn.LayerNorm(embedding_size)

        # output head
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        # GPT-2 Dropout
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_length)
        """
        batch_size, seq_len = x.shape

        # 构建位置序列 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        positions = positions.expand(batch_size, seq_len)  # [batch_size, seq_len]

        # 嵌入相加 + dropout
        tok_emb = self.token_embedding(x)  # (batch_size, seq_len, emb_size)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_len, emb_size)
        hidden_states = self.drop(tok_emb + pos_emb)

        # 依次输入 n 层TransformerBlock
        for block in self.layers:
            hidden_states = block(hidden_states, mask=mask)

        # 最终的LayerNorm
        hidden_states = self.ln_f(hidden_states)  # (batch_size, seq_len, emb_size)

        # 通过lm_head
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        return logits
    
    def generate_square_subsequent_mask(self, size):
        mask = torch.nn.Transformer.generate_square_subsequent_mask(size)
        return mask

def train(model, dataset, valid_dataset, num_epochs=3, batch_size=32, learning_rate=1.5e-4, device='cuda',
          max_length=128, warmup_ratio=0.03):
    model.to(device)
    model.train()

    # 配置日志输出到文件
    logging.basicConfig(
        filename="app.log",  # 指定日志文件路径
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s [%(levelname)s] %(message)s",  # 设置日志格式
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="a"  # 追加模式（默认），可选 "w" 表示覆盖模式
    )

    # Generate causal mask (causal attention mask) as a 2D matrix
    causal_mask = model.generate_square_subsequent_mask(max_length).to(device)  # Shape: [seq_length, seq_length]

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # calculate warmup steps
    total_steps = num_epochs * len(dataloader)
    warmup_steps = int(warmup_ratio * total_steps)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    scheduler = get_scheduler(
        name="cosine",  # 学习率调度类型
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,  # Warm-up 阶段的步数
        num_training_steps=total_steps  # 总训练步数
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # GradScaler for mixed precision
    scaler = GradScaler()

    # Total number of batches
    total_batches = len(dataloader)
    print("total batches num: ", total_batches)
    logging.info(f"total batches num: {total_batches}")
    valid_x = valid_dataset["input_ids"].to(device)
    valid_y = valid_dataset["input_ids"].to(device)

    valid_y = valid_y[:, 1:].contiguous()
    valid_y1 = valid_y.view(-1)

    # 每完成10%保存一次
    save_intervals = [int(total_batches * (i / 10)) for i in range(1, 11)]  # 保存点：[10%, 20%, ..., 100%]

    # Training loop
    for epoch in range(num_epochs):
        save_intervals_idx = 0  # 当前进度检查点索引
        total_loss = 0
        t1 = time.time()
        t2 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets from batch
            x = batch['input_ids'].to(device)  # Input token IDs
            y = batch['input_ids'].to(device)  # Target labels

            # Forward and backward pass with mixed precision
            optimizer.zero_grad()

            with autocast():  # Enable mixed precision
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

            # update learning rate
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"batch 10 index {batch_idx}: total_avg_loss {total_loss / (batch_idx + 1):.3f} current_loss {loss}")
                t3 = time.time()
                print(f"Time taken for 10 batches: {t3 - t2:.2f} sec\n")
                logging.info(
                    f"batch 10 index {batch_idx}: total_avg_loss {total_loss / (batch_idx + 1):.3f} current_loss {loss}\n"
                    f"Time taken for 10 batches: {t3 - t2:.2f} sec")
                t2 = time.time()

            if batch_idx % 400 == 0:
                with torch.no_grad():
                    outputs = model(valid_x, mask=causal_mask)

                    # Shift logits for causal language modeling
                    outputs = outputs[:, :-1, :].contiguous()
                    # Reshape outputs for calculating loss
                    outputs = outputs.view(-1, outputs.size(-1))

                    # Compute loss
                    loss = criterion(outputs, valid_y1)

                print(
                    f"batch 400 index {batch_idx}: valid_loss {loss}")
                t3 = time.time()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Current Learning Rate: {current_lr:.8f}")
                print(f"Time taken for evaluation: {t3 - t2:.2f} sec\n")
                logging.info(
                    f"batch 400 index {batch_idx}: valid_loss {loss}\n"
                    f"Current Learning Rate: {current_lr:.8f}\n"
                    f"Time taken for evaluation: {t3 - t2:.2f} sec")
                t2 = time.time()

            # Check if we need to save the model at this batch
            if save_intervals_idx < len(save_intervals) and (batch_idx + 1) == save_intervals[save_intervals_idx]:
                model_name = f'model2_{max_length}_{save_intervals_idx + 1}0_percent.pth'
                # SAVE PATH
                save_path = f'./{model_name}'
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at {save_path} after {save_intervals_idx + 1}0% of training.")
                logging.info(f"Model saved at {save_path} after {save_intervals_idx + 1}0% of training.")

                # 新增：打印平均损失值
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"Training progress: {save_intervals_idx + 1}0%, Average Loss so far: {avg_loss_so_far:.4f}")

                save_intervals_idx += 1  # Move to the next save interval

        # Print loss per epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Time taken for epoch: {time.time() - t1:.2f} sec\n")

    # # Save the model
    # # SAVE PATH
    # torch.save(model.state_dict(), './model.pth')


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
            generated_text.append(new_token)
            print(new_token, end="")

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    print("\n")
    return generated_text


# modified predict function
def predict_with_sampling(model, input_sequence, tokenizer, max_length=50, eos_token_id=None, device='cuda', top_k=10,
                          top_p=0.95, temperature=1.1):
    """
    Predict function using Top-K and Top-P sampling.
    """
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

            # Get the logits of the last token
            logits = outputs[:, -1, :]  # Extract logits for sampling
            logits = logits / temperature  # Adjust logits using temperature

            # Apply Top-K and Top-P sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (Top-P)
            sorted_indices_to_remove = cumulative_probs > top_p
            if top_k > 0:
                sorted_indices_to_remove[:, top_k:] = True  # Keep only Top-K tokens
            sorted_logits[sorted_indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probabilities = torch.softmax(sorted_logits, dim=-1)
            probabilities = probabilities.clamp(min=1e-9)  # make sure no zero division
            sampled_index_in_sorted = torch.multinomial(probabilities, num_samples=1)  # Sample index in the sorted list

            # Map the sampled index back to the original logits
            next_token = sorted_indices.gather(1, sampled_index_in_sorted)  # Map to original index

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

            # Decode the new token
            new_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated_text.append(new_token)
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
    max_length = 1024
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
