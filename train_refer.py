from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import get_scheduler
from model_refer import GPT2
import logging
import torch
from torch.nn import functional as F
import math
import time


@torch.no_grad()
def accuracy(logits, targets, pad_token_id=50256):
    prediction = torch.argmax(F.softmax(logits, dim=2), dim=2)

    # 生成 mask，忽略填充 token（mask 为 True 的地方会被计算）
    mask = targets.ne(pad_token_id)  # 不等于 pad_token_id 的地方为 True

    # 计算预测正确的情况
    correct = (prediction == targets) & mask  # 仅在 mask 位置计算

    # 计算准确率（忽略 padding）
    accuracy = correct.float().sum() / mask.float().sum()

    return accuracy.item()


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

    num_epochs = 1

    # 配置日志输出到文件
    logging.basicConfig(
        filename="app.log",  # 指定日志文件路径
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s [%(levelname)s] %(message)s",  # 设置日志格式
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="a"  # 追加模式（默认），可选 "w" 表示覆盖模式
    )

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    # dataset = load_from_disk("./bookcorpus_split_2")
    dataset = load_from_disk("./alpaca_1024")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    total_batches = len(dataloader)
    print("total batches num: ", total_batches)
    logging.info(f"total batches num: {total_batches}")
    # 每完成10%保存一次
    save_intervals = [int(total_batches * (i / 10)) for i in range(1, 11)]  # 保存点：[10%, 20%, ..., 100%]

    mixed = False
    dtype = 'float32'
    if not no_mixed:
        mixed = True
        dtype = 'float16'

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

    model.load_state_dict(torch.load('./GPT_512_100_percent.pth'))

    model.to(device)

    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=2.5e-4, betas=(0.9, 0.95),
                                           device_type=device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # learning rate scheduler
    scheduler = get_scheduler(
        name="cosine",  # 学习率调度类型
        optimizer=optimizer,
        num_warmup_steps=total_batches * warm_up,  # Warm-up 阶段的步数
        num_training_steps=total_batches  # 总训练步数
    )

    model.train()

    save_intervals_idx = 0
    total_loss = 0
    total_accuracy = 0

    for epoch in range(num_epochs):
        t1 = time.time()
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()

            x = data['input_ids'][..., :-1].contiguous().to(device)
            y = data['input_ids'][..., 1:].contiguous().to(device)
            if mixed:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()

            scheduler.step()  # 更新学习率    

            total_loss += loss.item()
            current_accuracy = accuracy(logits, y)
            total_accuracy += current_accuracy

            if batch_idx % 10 == 0 and batch_idx > 0:
                t2 = time.time()
                print(
                    f'Batch: {batch_idx}, Loss: {total_loss / (batch_idx + 1):.3f}, Accuracy: {total_accuracy / (batch_idx + 1):.4f}')
                print(f"Time taken for 10 batches: {t2 - t1:.2f} sec\n")
                logging.info(
                    f'Batch: {batch_idx}, Loss: {loss}, Accuracy: {current_accuracy}'
                    f'Time taken for 10 batches: {t2 - t1:.2f} sec\n')
                t1 = time.time()

            # Check if we need to save the model at this batch
            if save_intervals_idx < len(save_intervals) and (batch_idx + 1) == save_intervals[save_intervals_idx]:
                model_name = f'GPT_Alpaca_{max_length}_{save_intervals_idx + 1}0_percent.pth'
                # SAVE PATH
                save_path = f'./{model_name}'
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at {save_path} after {save_intervals_idx + 1}0% of training.")
                logging.info(f"Model saved at {save_path} after {save_intervals_idx + 1}0% of training.")

                # 新增：打印平均损失值
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"Training progress: {save_intervals_idx + 1}0%, Average Loss so far: {avg_loss_so_far:.4f}")

                save_intervals_idx += 1  # Move to the next save interval
