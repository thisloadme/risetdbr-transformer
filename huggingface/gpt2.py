from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "My favorite food is"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# print(len(output))
# print(len(output[0]))
# print(len(output[0][0]))
# exit()

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text) 