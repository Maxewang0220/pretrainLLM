from model import MyGPT2, predict

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
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import json


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


# def get_QA_token_prob(model, tokenizer, max_tokens=10, device='cuda', qa_data=qa_data):
#     model.to(device)
#     model.eval()
#
#     # get a random question and answer pair
#     random_qa = random.choice(qa_data)
#     question = random_qa["question"]
#     real_answer = random_qa["answer"]
#     print("question is : ", question)
#     print("real_answer is : ", real_answer)
#
#     answer_tokens = tokenizer(real_answer, truncation=True, max_length=200, return_tensors="pt")["input_ids"].to(
#         device)
#     print("answer_tokens: ", answer_tokens[0])
#     print("shape of answer_tokens: ", answer_tokens.shape)
#
#     # 假设 answer_tokens 是通过 tokenizer 得到的 input_ids
#     answer_tokens_ids = answer_tokens[0].tolist()  # 转换为列表
#     tokens = tokenizer.convert_ids_to_tokens(answer_tokens_ids)  # 转换为 token
#     print("Tokens: ", tokens)
#
#     # Tokenize input sentence and move to device
#     input_sequence = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
#     generated_sequence = input_sequence.clone()
#
#     token_distributions = []  # Store probability distributions for each generated token
#
#     with torch.no_grad():
#         for _ in range(max_tokens):
#             # Forward pass to get logits
#             outputs = model(generated_sequence)
#             logits = outputs[:, -1, :]  # Get the logits for the last token in the sequence
#
#             # Convert logits to probabilities
#             probs = softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
#
#             # Store the probability distribution
#             token_distributions.append(probs[0].cpu().numpy())
#
#             # Get the next token (argmax or sampling, here using argmax)
#             next_token = torch.argmax(probs, dim=-1, keepdim=True)
#
#             # Append the predicted token to the sequence
#             generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
#
#     # convert to numpy array
#     # convert to numpy array
#     token_distributions_array = np.array(token_distributions)
#     print(token_distributions_array.shape)  # 输出 (10, 50257)
#
#     # Convert answer_tokens to CPU and NumPy array
#     answer_tokens = answer_tokens.cpu().numpy()  # 转为 NumPy 数组
#     print("answer_tokens: ", answer_tokens)  # 输出 answer_tokens:  [464]
#     print("len(answer_tokens): ", len(answer_tokens))  # 输出 len(answer_tokens):  1
#
#     # get the probability of the target token
#     selected_probs = token_distributions_array[:, answer_tokens]  # Shape: (max_tokens, len(answer_tokens))
#     print("Selected probabilities:\n", selected_probs)
#     print("Shape: ", selected_probs.shape)
#
#     return token_distributions


# model.to(device)
# https://huggingface.co/thanhnew2001/everything

# 3nd part of the evaluation


# gpt 给的

def get_QA_token_prob(model, tokenizer, qa_data, max_tokens=10, device='cuda'):
    """
    计算真实答案 `real_answer` 中 token 在模型生成 token 分布中的概率。

    参数：
    - model: 语言模型 (GPT2)
    - tokenizer: 与模型匹配的 tokenizer
    - qa_data: 问答数据集 (包含 question 和 answer)
    - max_tokens: 最大生成 token 数
    - device: 运行设备 ('cuda' 或 'cpu')

    返回：
    - token_distributions: 存储每一步 token 概率分布的列表
    """
    model.to(device)
    model.eval()

    # 1️⃣  随机选择一个 Q&A
    random_qa = random.choice(qa_data)
    question = random_qa["question"]
    real_answer = random_qa["answer"]

    print(f"Question: {question}")
    print(f"Real Answer: {real_answer}")

    # 2️⃣ Tokenize real answer
    answer_tokens = tokenizer(real_answer, truncation=True, max_length=200, return_tensors="pt")["input_ids"].to(device)
    answer_token_ids = answer_tokens[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(answer_token_ids)
    print(f"Tokens: {tokens}")

    # 3️⃣ Tokenize question
    input_sequence = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
    generated_sequence = input_sequence.clone()

    token_distributions = []  # 存储每个生成 token 的概率分布

    with torch.no_grad():
        for _ in range(max_tokens):
            # 获取模型 logits
            logits, _ = model(generated_sequence)  # 你的 GPT2 forward 返回 logits, loss

            # 取最后一个 token 的 logits
            logits = logits[:, -1, :]

            # 计算 softmax 概率
            probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)

            # 存储概率分布
            token_distributions.append(probs[0].cpu().numpy())

            # 选择下一个 token（改为随机采样）
            next_token = torch.multinomial(probs, num_samples=1)

            # 将预测的 token 添加到序列
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

    # 4️⃣ 计算真实答案的 token 在生成概率中的位置
    token_distributions_array = np.array(token_distributions)  # (max_tokens, vocab_size)

    print(f"Token Distributions Shape: {token_distributions_array.shape}")  # (max_tokens, vocab_size)

    # 5️⃣ 获取真实答案 token 的概率
    answer_tokens_cpu = answer_tokens.cpu().numpy().flatten()
    min_len = min(len(token_distributions), len(answer_tokens_cpu))

    selected_probs = token_distributions_array[np.arange(min_len), answer_tokens_cpu[:min_len]]

    print(f"Selected Probabilities:\n {selected_probs}")
    print(f"Shape of Selected Probabilities: {selected_probs.shape}")

    return token_distributions


def generate_write_n_sentences(model, tokenizer, device='cuda', num_sentence=10):
    model.to(device)
    model.eval()

    rate_prompt = """You are a language expert tasked with evaluating a set of generated sentences. Please rate each sentence based on the following criteria:
Grammar and Syntax (1-10): Does the sentence follow proper grammar and syntax rules?
Semantic Clarity (1-10): Is the meaning of the sentence clear and easy to understand?
Contextual Relevance (1-10): Is the sentence relevant to the given topic or theme?
Creativity and Style (1-10): Does the sentence demonstrate creativity or an appropriate style?
For each sentence, provide a detailed score (1-10) for each category, along with a brief explanation for your ratings.

Here are the sentences to evaluate:

[Sentence 1]
[Sentence 2]
[Sentence 3]
Please respond in the following format:

Sentence 1:

Grammar and Syntax: X/10
Semantic Clarity: X/10
Contextual Relevance: X/10
Creativity and Style: X/10
Reasoning: [Provide a detailed explanation]"""

    # load the CNN/DailyMail dataset as prompt
    cnn_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)

    # write to a file
    with open("generated_sentences.txt", "w") as f:
        f.write(rate_prompt)
        f.write("Generated sentences:\n")
        for i in range(num_sentence):
            sentence_idx = "This is sentence " + str(i) + ":\n"
            # original text
            input_text = cnn_dataset[i]["article"]
            print("Context: ", input_text, '\n')
            # tokenize and truncate to max_length tokens
            # max_length=200, generate a lot ( and .
            # encoded = tokenizer(input_text, truncation=True, max_length=200, return_tensors="pt")
            encoded = tokenizer(input_text, truncation=True, max_length=20, return_tensors="pt")

            # decode the tokenized input text
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            print("input_text: ", input_text, '\n')

            # string
            print("Generated text: ")
            generated_text = predict(model, input_text, tokenizer, max_length=50, eos_token_id=tokenizer.eos_token_id,
                                     device='cuda')
            print("\n")
            # char lists to string sentence
            generated_text = ''.join(generated_text).strip()

            f.write(sentence_idx)
            f.write(generated_text + "\n")


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
    model.load_state_dict(torch.load("GPT_512_50_2_percent.pth"))
    model.eval()  # ============================================

    # 1->perplexity
    # 2->QA token prob
    # 3->generate and write n sentences
    evaluate_mode = 2

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

        get_QA_token_prob(model, tokenizer, max_tokens=10, device=device)
    # generate and write n sentences
    elif evaluate_mode == 3:
        generate_write_n_sentences(model, tokenizer, device, num_sentence=10)
    else:
        print("Invalid evaluation mode. Please choose 1, 2, or 3.")
