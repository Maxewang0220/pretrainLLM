import datasets
from transformers import GPT2Tokenizer


# Load the pretraining dataset
def load_dataset(dataset_name, split):
    return datasets.load_dataset(
        dataset_name,
        split=split
    )


# tokenize corpus transfer text 2 tokens
def tokenize_corpus(dataset, tokenizer, max_length=512):
    # Set pad token to eos token if pad token is None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    # transfer text into token_ids
    tokens = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,  # 截断长文本，保证最大长度不超过 max_length
            padding="max_length",  # 使用固定长度填充, 生成attention_mask
            max_length=max_length  # 最大长度
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
    pretraining_dataset = load_dataset(
        "upstage/Pretraining_Dataset",
        split="train"
    )

    print(pretraining_dataset)
    for example in pretraining_dataset["text"][:5]:
        print(example)
