import datasets


# Load the pretraining dataset
# def load_dataset(dataset_name, split):
#     dataset = datasets.load_dataset(
#         dataset_name,
#         split=split
#     )
#
#     filtered_dataset = dataset.filter(
#         lambda example: example['meta'].get("redpajama_set_name") not in ["RedPajamaGithub", "RedPajamaArXiv",
#                                                                           "RedPajamaStackExchange"],
#         batched=False
#     )
#
#     return filtered_dataset

def load_dataset(dataset_name, split, trust_remote_code=True):
    # 加载数据集
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=trust_remote_code
    )

    # 过滤掉 source 为 "github" 且长度小于 512 的数据
    filtered_dataset = dataset.filter(
        lambda example: example.get('source') != "github" and len(example.get('text', '')) >= 512,
        batched=False
    )

    return filtered_dataset


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
    pretraining_dataset = load_dataset(
        "upstage/Pretraining_Dataset",
        split="train"
    )

    print(pretraining_dataset)
    for example in pretraining_dataset["text"][:5]:
        print(example)
