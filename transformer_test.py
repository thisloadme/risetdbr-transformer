import torch
import torch.nn as nn
import os
from transformer_pytorch import generate_square_subsequent_mask, get_batch, batchify, data_process, TransformerModel, decode_tensor
# print('asd')
# exit()
bptt = 35
text_data = "bagaimana cara membayar fidyah"
tensor_data = data_process([text_data])

with open('dataset.txt', 'r', encoding='utf-8') as f:
    dataset = f.read().lower()
vocabulary = sorted(list(set(dataset)))
# print(vocabulary)
# output_argmax = [34, 43, 30, 43,  1, 30, 43, 54, 43, 42, 34, 43, 30, 43, 42, 30, 47, 30, 34, 43, 30, 43, 30, 42, 38,  1, 30, 30, 43]
# words = [vocabulary[i.item()] for i in output_argmax]
# print(words)
# exit()

data = torch.load(os.path.join(os.getcwd(), "model_data.pth"))
model_state = data['model_state']
ntokens = data['ntokens']
emsize = data['emsize']
nhead = data['nhead']
d_hid = data['d_hid']
nlayers = data['nlayers']
dropout = data['dropout']
bptt = data['bptt']
device = data['device']

src_mask = generate_square_subsequent_mask(bptt).to(device)

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.load_state_dict(model_state)
model.eval()

data, targets = get_batch(tensor_data, 0)
output = model(data, src_mask)

softmax = nn.Softmax(dim=2)
# print(tensor_data)
# print(output)
print(tensor_data.shape)
print(output.shape)
# print(torch.argmax(torch.abs(output), dim=-1))
print('-' * 10)
# print(torch.argmax(output, dim=2))
out_softmax = softmax(output)
print(out_softmax)
print('-' * 10)
# print(torch.argmax(out_softmax, dim=2) == torch.argmax(output, dim=2))

a = []
for item in torch.argmax(out_softmax, dim=2):
    print(item)
    # for i in item:
    #     a.append(vocabulary[i.item()])
# words = [vocabulary[i.item()] for item in torch.argmax(torch.abs(output), dim=2)]
# print(words)
print(''.join(a))
exit()