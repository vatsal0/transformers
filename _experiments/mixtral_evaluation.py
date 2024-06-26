from transformers import AutoTokenizer, MixtralForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/new_proj64_soft2/checkpoint-55000'
device = 'cuda'

model = MixtralForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')

# test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='/fs/nexus-scratch/vatsalb/huggingface')
# encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')
# torch.save(encodings.input_ids, 'wikitext2_test_tokens.pt')

encodings = {'input_ids': torch.load('wikitext2_test_tokens.pt')}

max_length = 1024
stride = 1024
seq_len = encodings['input_ids'].size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
  end_loc = min(begin_loc + max_length, seq_len)
  trg_len = end_loc - prev_end_loc
  input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
  target_ids = input_ids.clone()
  target_ids[:, :-trg_len] = -100
  
  with torch.no_grad():
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
  
  if not torch.isnan(neg_log_likelihood):
    nlls.append(neg_log_likelihood)
  
  prev_end_loc = end_loc
  if end_loc == seq_len:
    break

perplexity = torch.exp(torch.stack(nlls).mean())

print(f'{perplexity=}')
