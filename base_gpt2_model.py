from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# input text
input_text = "A spectre is haunting Europe — the spectre of"

# encode text
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# generate text
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2)

# decode text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
