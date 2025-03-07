import datasets
from datasets import Dataset
from transformers import GPT2Tokenizer


# Load the pretraining dataset
def load_dataset_bookcorpus(dataset_name, split, tokenizer, max_length=512, concat_size=20480):
    # Step 1: Load dataset
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True
    )

    # Step 2: Concatenate text
    def concatenate_text(examples):
        # Split the text into batches of `concat_size` paragraphs
        concatenated_text = []
        buffer = []
        for text in examples["text"]:
            buffer.append(text)
            if len(buffer) >= concat_size:  # If buffer reaches `concat_size`, join and append
                concatenated_text.append("".join(buffer))
                buffer = []
        if buffer:  # Add remaining text if any
            concatenated_text.append("".join(buffer))

        return {"text": concatenated_text}

    concatenated_dataset = dataset.map(
        concatenate_text,
        batched=True,
        batch_size=1000,  # Adjust depending on memory
        num_proc=16  # Number of parallel processes
    )

    # Step 3: Chunk tokenized text
    def tokenize_and_chunk(examples):
        text = examples["text"]
        tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]

        chunks = []

        # Slice with step_size=chunk_size
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            # Ensure padding only happens for the last chunk
            if len(chunk) == max_length:
                chunks.append(chunk)
            elif len(chunk) > max_length * 0.8:
                chunk += [tokenizer.eos_token_id] * (max_length - len(chunk))
                chunks.append(chunk)

        return {"input_ids": chunks}

    chunked_dataset = concatenated_dataset.map(
        tokenize_and_chunk,
        batched=False
    )

    # Flatten all chunks and create new columns for input_ids and labels
    flattened_input_ids = [chunk for chunks in chunked_dataset["input_ids"] for chunk in chunks]

    # Create a new dataset with input_ids and labels
    new_dataset = Dataset.from_dict({"input_ids": flattened_input_ids})

    # Set the dataset format to PyTorch
    new_dataset.set_format(type="torch", columns=["input_ids"])

    new_dataset.save_to_disk("./bookcorpus_split")

    return new_dataset


# Load the Alpaca dataset
def load_dataset_Alpaca(dataset_name, split, tokenizer, max_length=512):
    # Step 1: Load dataset
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True
    )

    # Step 2: Concatenate text
    def concatenate_text(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            # 如果 input 为空，则只拼接 instruction 和 output
            if inp and inp.strip():
                full_text = f"{instr}\n{inp}\n{out}"
            else:
                full_text = f"{instr}\n{out}"
            texts.append(full_text)
        return {"text": texts}

    concatenated_dataset = dataset.map(
        concatenate_text,
        batched=True,
        batch_size=1000,  # Adjust depending on memory
        num_proc=1,  # Number of parallel processes
    )

    # Step 3: Chunk tokenized text
    def tokenize_and_pad(examples):
        text = examples["text"]
        tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]

        # 如果超过最大长度，则截断；不足则用 eos_token_id 填充
        if len(tokens) >= max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [tokenizer.eos_token_id] * (max_length - len(tokens))
        return {"input_ids": tokens}

    tokenized_dataset = concatenated_dataset.map(
        tokenize_and_pad,
        batched=False
    )

    # Step 4: 设置为 PyTorch 格式，并保存预处理后的数据
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_dataset.save_to_disk("./alpaca_512")

    return tokenized_dataset


if __name__ == "__main__":
    # pretraining_dataset = load_dataset(
    #     "upstage/Pretraining_Dataset",
    #     split="train"
    # )
    #
    # print(pretraining_dataset)
    # for example in pretraining_dataset["text"][:5]:
    #     print(example)

    load_dataset_Alpaca("tatsu-lab/alpaca", split="train", tokenizer=GPT2Tokenizer.from_pretrained("gpt2"))
