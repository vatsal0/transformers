from transformers import AutoTokenizer, MixtralForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_dropout0.4/checkpoint-35000'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_final/checkpoint-35000' # cluster 63
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_dropout0.4_router0.001_inverse_eps_proj16/checkpoint-27500'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/new_proj64_soft2/checkpoint-55000'

'''
router0.01 138
router0.005 146 
router0.001 133

negative proj8 138
inverse_eps proj8 136

inverse_eps proj16 162 (with init)

proj16 beta 0 162

proj16 beta 0.4 184
proj16 beta 0.2 212

proj32 beta 0.1 139
proj16 beta 0.1 142
proj4 beta 0.1 148
'''
# model.model.layers[0].block_sparse_moe.gate.weight[0].norm()
# model.model.layers[0].block_sparse_moe.gate.embed.T[0].norm()
device = 'cuda'

model = MixtralForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')

# test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='/fs/nexus-scratch/vatsalb/huggingface')
# encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')
encodings = {'input_ids': torch.load('wikitext2_test_tokens.pt')}

max_length = 1024
stride = 1024
seq_len = encodings['input_ids'].size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
  end_loc = min(begin_loc + max_length, seq_len)
  trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
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
