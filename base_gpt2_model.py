from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# decode text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

