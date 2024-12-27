# Import required libraries
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)
print(indexed_tokens)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(device)
model.to(device)

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
    print(predictions)

# Get the predicted next sub-word
predicted_index = torch.argmax(predictions[0, -1, :]).item()
print(predicted_index)
exit()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# Print the predicted word
print(predicted_text)