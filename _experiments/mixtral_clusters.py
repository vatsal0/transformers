from transformers import AutoTokenizer, MixtralForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inv/checkpoint-12500'
device = 'cuda'

model = MixtralForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')

test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')


'''
even with router coef 0.1 expert distributions collapse. need detach
'''

text = '''
He moved on in the summer of 759 ; this has traditionally been ascribed to famine , but Hung believes that frustration is a more likely reason . He next spent around six weeks in Qinzhou ( now Tianshui , Gansu province ) , where he wrote more than sixty poems .
'''

# text = 'He wrote more than sixty poems.'

# text=test['text'][100]

def hook(module, input, output):
  # print(module)
  # print(input[0].shape)
  # print(output.shape)
  # print(input)
  print(output)
  print(torch.nn.functional.softmax(output, dim=1))

gate_hook = model.model.layers[-4].block_sparse_moe.gate.register_forward_hook(hook)

encodings = tokenizer(text, return_tensors='pt').to(device)
outputs = model(encodings.input_ids)

gate_hook.remove()

# model.model.layers[-1].block_sparse_moe.gate.embed

