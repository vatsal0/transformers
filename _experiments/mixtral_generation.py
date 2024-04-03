from transformers import AutoTokenizer, AutoConfig, MixtralForCausalLM

MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_dropout0.4/checkpoint-35000'

device='cuda'

model = MixtralForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')
tokenizer.pad_token = tokenizer.eos_token

def gen(text):
  input = tokenizer(text, return_tensors='pt').to(device)
  
  generate_ids = model.generate(**input, max_length=50, do_sample=True)
  return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

