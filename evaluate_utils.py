from model_refer import GPT2
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np


def calculate_perplexity(model, dataloader, device='cuda'):
    """
    to calculate the mean perplexity of the dataset
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # tqdm
        for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
            input_ids = batch["input_ids"].to(device)

            # logits
            logits, _ = model(input_ids, targets=input_ids)

            # cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=50256,  # ignore the padding token and eos token
                reduction='sum'
            )

            total_loss += loss.item()  # total loss
            total_tokens += input_ids.numel()  # number of tokens in the batch

    # average loss
    avg_loss = total_loss / total_tokens
    mean_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return mean_perplexity


def get_QA_dataset_avg_prob(model, tokenizer, qa_data, device='cuda'):
    """
    to calculate the average probability of all Q&As
    """
    model.to(device)
    model.eval()

    qa_avg_probs = []  # to store the average probabilities of each Q&A

    for idx, qa_pair in enumerate(qa_data):

        question = qa_pair["question"].strip() + " "  # get rid of extra spaces
        real_answer = qa_pair["answer"]

        print(f"\nProcessing Q&A {idx + 1}/{len(qa_data)}")
        print(f"Q: {question}")
        print(f"A: {real_answer}")

        # Tokenize the answer
        answer_tokens = tokenizer(real_answer, return_tensors="pt")["input_ids"].to(device)
        answer_token_ids = answer_tokens[0].tolist()

        # Get tokens of the answer
        tokens = tokenizer.convert_ids_to_tokens(answer_token_ids)
        print(f"Tokens: {tokens}")

        # tokenize the question
        input_sequence = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
        # to store the probabilities of each token of the answer
        selected_probs = []  # store the probabilities of each token of the answer

        # step by step, calculate the probability of each "right" token
        for i, token_id in enumerate(answer_token_ids):
            with torch.no_grad():
                logits, _ = model(input_sequence)
                probs = F.softmax(logits, dim=-1)
                # get the probabilities of the next "right" token
                token_prob = probs[0, token_id].item()
                selected_probs.append(token_prob)

            print(f"Step {i + 1}: P('{tokens[i]}') = {token_prob:.9f}")

            # add the token to the input sequence
            input_sequence = torch.cat((input_sequence, torch.tensor([[token_id]], device=device)), dim=1)
            print(f"Updated input sequence: {tokenizer.decode(input_sequence[0])}")
        # to calculate the average probability of the answer
        qa_avg_prob = np.mean(selected_probs) if selected_probs else 0
        print(f"Average probability for this Q&A: {qa_avg_prob:.9f}")

        qa_avg_probs.append(qa_avg_prob)

    # avearge probability of the dataset
    dataset_avg_prob = np.mean(qa_avg_probs) if qa_avg_probs else 0
    print(f"\nFinal Dataset Average Probability: {dataset_avg_prob:.9f}")

    return dataset_avg_prob


def generate_write_n_sentences(model, tokenizer, device, num_sentence=10, max_new_tokens=200, temperature=0.8,
                               top_k=40):
    model.eval()

    rating_prompt = """You are a language expert tasked with evaluating a set of generated sentences. Please rate each sentence based on the following criteria:
    Grammar and Syntax (1-10): Does the sentence follow proper grammar and syntax rules?
    Semantic Clarity (1-10): Is the meaning of the sentence clear and easy to understand?
    Contextual Relevance (1-10): Is the sentence relevant to each other?
    Creativity and Style (1-10): Does the sentence demonstrate creativity or an appropriate style?
    For each sentence, provide a detailed score (1-10) for each category.

    Please respond in the following format:

    Sentence 1:

    Grammar and Syntax: X/10
    Semantic Clarity: X/10
    Contextual Relevance: X/10
    Creativity and Style: X/10 """

    generated_sentences = []
    generated_sentences.append(rating_prompt + "\n")  # 添加提示语

    for i in range(num_sentence):
        # start with a start token
        start_token = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

        # to generate
        with torch.no_grad():
            generated_tokens = model.generate(
                idx=start_token,
                max_new_tokens=max_new_tokens,
                determined=False,
                temperature=temperature,
                top_k=top_k
            )

        # decode
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

        # get rid of special tokens at the beginning
        generated_text = generated_text.lstrip(",.:;!?")  # delete
        generated_sentences.append(f"\nThis is sentence {i + 1}:")  # number the sentences
        generated_sentences.append(generated_text)

    # write
    with open("generated_sentences.txt", "w", encoding="utf-8") as f:
        for sentence in generated_sentences:
            f.write(sentence + "\n\n")

    print(f"Generated {num_sentence} sentences and saved to 'generated_sentences.txt'.")


if __name__ == '__main__':
    no_mixed = False
    batch_size = 16
    vocab_size = 50257
    max_length = 512
    num_layers = 6  # 6 for GPT_Base_512_50_percent & GPT_Base_512_50_percent; 12 for others
    num_heads = 6  # 6 for GPT_Base_512_50_percent & GPT_Base_512_50_percent; 12 for others
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
    model.load_state_dict(torch.load("GPT_Base_512_50_percent.pth"))
    model.eval()

    # 1->perplexity
    # 2->QA token prob
    # 3->generate and write n sentences
    evaluate_mode = 2

    # perplexity evaluation
    if evaluate_mode == 1:
        tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset("stas/openwebtext-10k", split="train")


        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])

        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        # calculate the mean perplexity of the dataset
        mean_ppl = calculate_perplexity(model, dataloader, device=device)
        print(f"Dataset Perplexity: {mean_ppl}")

    # QA token prob
    elif evaluate_mode == 2:
        with open("qa_dataset.json", "r", encoding="utf-8") as file:
            qa_data = json.load(file)

        get_QA_dataset_avg_prob(model, tokenizer, device=device, qa_data=qa_data)
    # generate and write n sentences
    elif evaluate_mode == 3:
        # generate 10 sentences and write them to a file
        generate_write_n_sentences(model, tokenizer, device, num_sentence=10)
    else:
        print("Invalid evaluation mode. Please choose 1, 2, or 3.")
