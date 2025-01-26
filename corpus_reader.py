import datasets
from datasets import Dataset
from transformers import GPT2Tokenizer


# Load the pretraining dataset
def load_dataset(dataset_name, split, tokenizer, max_length=128):
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True
    )

    def tokenize_and_chunk(example):
        # tokenize the text
        text = example["text"]
        tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]

        # split the tokens into chunks of max_length and pad the last chunk if needed
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]

            # Ensure padding only happens for the last chunk
            if len(chunk) == max_length:
                chunks.append(chunk)
            elif len(chunk) > max_length * 0.8:
                chunk += [tokenizer.eos_token_id] * (max_length - len(chunk))
                chunks.append(chunk)

        return {"input_ids": chunks, "labels": chunks}

    chunked_dataset = dataset.map(tokenize_and_chunk, batched=False)

    # Flatten all chunks and create new columns for input_ids and labels
    flattened_input_ids = [chunk for chunks in chunked_dataset["input_ids"] for chunk in chunks]
    flattened_labels = [chunk for chunks in chunked_dataset["labels"] for chunk in chunks]

    # Create a new dataset with input_ids and labels
    new_dataset = Dataset.from_dict({"input_ids": flattened_input_ids, "labels": flattened_labels})

    # Set the dataset format to PyTorch
    new_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return new_dataset


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

    print("save successfully")

    return new_dataset


def load_dataset_wiki(split, tokenizer, max_length=128):
    # 加载Wikipedia英文数据集
    dataset = datasets.load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=split
    )

    def tokenize_and_chunk(example):
        # tokenize the text
        tokens = tokenizer(example["text"], truncation=False, padding=False)["input_ids"]

        # split the tokens into chunks of max_length and pad the last chunk if needed
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]

            # Ensure padding only happens for the last chunk
            if len(chunk) < max_length:
                chunk += [tokenizer.eos_token_id] * (max_length - len(chunk))

            chunks.append(chunk)

        return {"input_ids": chunks, "labels": chunks}

    chunked_dataset = dataset.map(tokenize_and_chunk, batched=False)

    # Flatten all chunks and create new columns for input_ids and labels
    flattened_input_ids = [chunk for chunks in chunked_dataset["input_ids"] for chunk in chunks]
    flattened_labels = [chunk for chunks in chunked_dataset["labels"] for chunk in chunks]

    # Create a new dataset with input_ids and labels
    new_dataset = Dataset.from_dict({"input_ids": flattened_input_ids, "labels": flattened_labels})

    # Set the dataset format to PyTorch
    new_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return new_dataset


# tokenize corpus transfer text 2 tokens
def tokenize_corpus(dataset, tokenizer, max_length=128):
    # Set pad token to eos token if pad token is None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    # transfer text into token_ids
    tokens = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,  # clip long texts
            padding="max_length",  # padding with fixed length
            max_length=max_length
        ),
        batched=True,
    )

    # Add labels for language modeling (labels = input_ids)
    tokens = tokens.map(
        lambda x: {"labels": x["input_ids"]},
        batched=True
    )

    # Convert to PyTorch tensors
    tokens.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokens


if __name__ == "__main__":
    # pretraining_dataset = load_dataset(
    #     "upstage/Pretraining_Dataset",
    #     split="train"
    # )
    #
    # print(pretraining_dataset)
    # for example in pretraining_dataset["text"][:5]:
    #     print(example)

    dataset = load_dataset_wiki("train[:1]", GPT2Tokenizer.from_pretrained('gpt2'), 128)
    print(type(dataset["input_ids"]))
