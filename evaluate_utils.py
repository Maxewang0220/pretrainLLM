from Q_A import qa_data
from model_refer import GPT2

# 导入 Transformers 库的 GPT2 Tokenizer
from transformers import GPT2Tokenizer

# 导入 PyTorch 相关库
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import json
# 批量
import torch
import torch.nn.functional as F
import numpy as np


def calculate_perplexity(model, dataloader, device='cuda'):
    """
    计算整个数据集的平均 perplexity (PPL) 并显示进度条
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # tqdm 进度条，显示 DataLoader 进度
        for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
            input_ids = batch["input_ids"].to(device)  # (batch_size, seq_len)

            # 获取 logits
            logits, _ = model(input_ids, targets=input_ids)  # (batch_size, seq_len, vocab_size)

            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
                input_ids.view(-1),  # (batch_size * seq_len)
                ignore_index=50256,  # 忽略 GPT-2 的 <|endoftext|>
                reduction='sum'  # 计算总损失
            )

            total_loss += loss.item()  # 累加损失
            total_tokens += input_ids.numel()  # 计算总 token 数

    # 计算平均 loss
    avg_loss = total_loss / total_tokens
    mean_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return mean_perplexity


def get_QA_dataset_avg_prob(model, tokenizer, qa_data, device='cuda'):
    """
    计算整个问答数据集的平均概率
    """
    model.to(device)
    model.eval()

    qa_avg_probs = []  # 存储每个 Q&A 的平均概率

    for idx, qa_pair in enumerate(qa_data):
        # question = qa_pair["question"] + " "
        question = qa_pair["question"].strip() + " "  # ✅ 确保只有一个空格

        real_answer = qa_pair["answer"]

        print(f"\nProcessing Q&A {idx + 1}/{len(qa_data)}")
        print(f"Q: {question}")
        print(f"A: {real_answer}")

        # 1️⃣ Tokenize 答案
        answer_tokens = tokenizer(real_answer, return_tensors="pt")["input_ids"].to(device)
        answer_token_ids = answer_tokens[0].tolist()

        # 获取 token 名称
        tokens = tokenizer.convert_ids_to_tokens(answer_token_ids)
        print(f"Tokens: {tokens}")

        # 2️⃣ Tokenize 问题
        input_sequence = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
        selected_probs = []  # 存储当前 Q&A 的每个 token 概率

        # 3️⃣ 逐步计算每个 token 在 softmax 分布中的概率
        for i, token_id in enumerate(answer_token_ids):
            with torch.no_grad():
                # 根据您的模型实现，forward返回(logits, loss)
                logits, _ = model(input_sequence)

                # 由于您的模型在预测模式下直接返回最后一个位置的logits
                # 所以logits已经是[batch_size, vocab_size]的形状
                probs = F.softmax(logits, dim=-1)  # 不需要索引[-1]
                token_prob = probs[0, token_id].item()  # 获取当前 token 的概率
                selected_probs.append(token_prob)

            print(f"Step {i + 1}: P('{tokens[i]}') = {token_prob:.9f}")

            # 将当前 token 追加到输入序列，以预测下一个 token
            input_sequence = torch.cat((input_sequence, torch.tensor([[token_id]], device=device)), dim=1)
            print(f"Updated input sequence: {tokenizer.decode(input_sequence[0])}")
        # 4️⃣ 计算当前 Q&A 的平均 token 概率
        qa_avg_prob = np.mean(selected_probs) if selected_probs else 0
        print(f"Average probability for this Q&A: {qa_avg_prob:.9f}")

        qa_avg_probs.append(qa_avg_prob)

    # 5️⃣ 计算所有 Q&A 的平均概率的平均值
    dataset_avg_prob = np.mean(qa_avg_probs) if qa_avg_probs else 0
    print(f"\nFinal Dataset Average Probability: {dataset_avg_prob:.9f}")

    return dataset_avg_prob


def generate_write_n_sentences(model, tokenizer, device, num_sentence=10, max_new_tokens=200, temperature=0.8,
                               top_k=40):
    model.eval()
    generated_sentences = []

    for i in range(num_sentence):
        # 确保使用适当的起始 token
        start_token = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

        # 生成文本
        with torch.no_grad():
            generated_tokens = model.generate(
                idx=start_token,
                max_new_tokens=max_new_tokens,
                determined=False,
                temperature=temperature,
                top_k=top_k
            )

        # 解码文本
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

        # 确保句子不会有无意义的前缀
        generated_text = generated_text.lstrip(",.:;!?")  # 删除前导符号
        generated_sentences.append(f"\nThis is sentence {i + 1}:")  # 添加句子编号
        generated_sentences.append(generated_text)

    # 写入文件
    with open("generated_sentences.txt", "w", encoding="utf-8") as f:
        for sentence in generated_sentences:
            f.write(sentence + "\n")

    print(f"Generated {num_sentence} sentences and saved to 'generated_sentences.txt'.")


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
    model.load_state_dict(torch.load("GPT_Alpaca_512_100_percent.pth"))
    model.eval()  # ============================================

    # 1->perplexity
    # 2->QA token prob
    # 3->generate and write n sentences
    evaluate_mode = 3

    # perplexity evaluation
    if evaluate_mode == 1:
        tokenizer.pad_token = tokenizer.eos_token  # 解决 padding 问题
        # 2️⃣ 加载数据集
        dataset = load_dataset("stas/openwebtext-10k", split="train")


        # 3️⃣ Tokenization 处理
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 4️⃣ 转换格式
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

        # 创建 DataLoader
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        # 计算数据集的平均 Perplexity
        mean_ppl = calculate_perplexity(model, dataloader, device=device)
        print(f"Dataset Perplexity: {mean_ppl}")

    # QA token prob
    elif evaluate_mode == 2:

        # 读取 JSON 文件
        with open("qa_dataset.json", "r", encoding="utf-8") as file:
            qa_data = json.load(file)

        # 打印数据集内容
        print(json.dumps(qa_data, indent=4))

        get_QA_dataset_avg_prob(model, tokenizer, device=device, qa_data=qa_data)
    # generate and write n sentences
    elif evaluate_mode == 3:
        # 生成 10 句话并保存到文件
        generate_write_n_sentences(model, tokenizer, device, num_sentence=10)
    else:
        print("Invalid evaluation mode. Please choose 1, 2, or 3.")
