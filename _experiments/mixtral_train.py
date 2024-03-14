from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import wandb

with open('wandbkey') as f:
  key = f.read()

wandb.login(key=key)
run = wandb.init(
  project='Mistral', 
  job_type='training', 
  anonymous='allow'
)

ATTENTION_DROPOUT = 0.4
CLUSTER_EXPERTS=False
LEARNING_RATE = 4e-5

SAVE_STEPS=2000

OUTPUT_PATH='/fs/nexus-scratch/vatsalb/mixtral/'
OUTPUT_DIR=('cluster' if CLUSTER_EXPERTS else 'original')+f'_dropout{ATTENTION_DROPOUT}'

device='cuda'

config = MixtralConfig(
  vocab_size=32000,
  hidden_size=4096 // 4,
  intermediate_size=14336 // 4,
  num_hidden_layers=32 // 4,
  num_attention_heads=32 // 2,
  num_key_value_heads=8,
  hidden_act='silu',
  max_position_embeddings=4096 * 32 // 32,
  initializer_range=0.02,
  rms_norm_eps=1e-5,
  use_cache=True,
  pad_token_id=None,
  bos_token_id=1,
  eos_token_id=2,
  tie_word_embeddings=False,
  rope_theta=1e6,
  sliding_window=None,
  attention_dropout=ATTENTION_DROPOUT,
  num_experts_per_tok=2,
  num_local_experts=8 // 2,
  output_router_logits=True, #False,
  router_aux_loss_coef=0.001,
  cluster_experts=CLUSTER_EXPERTS
)
model = MixtralForCausalLM(config).to(device)

print(f'{ATTENTION_DROPOUT=} {LEARNING_RATE=}')
print(f'{sum(p.numel() for p in model.parameters())} Parameters')

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

train_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='train').shuffle()
eval_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='validation')

training_arguments = TrainingArguments(
    output_dir=OUTPUT_PATH + OUTPUT_DIR,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    # eval_accumulation_steps=2,
    optim='paged_adamw_32bit',
    save_steps=SAVE_STEPS,
    logging_steps=500,
    save_total_limit=5,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    # max_steps=2000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='linear',
    evaluation_strategy='steps',
    eval_steps=SAVE_STEPS,
    report_to='wandb'
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length= 1024,
    dataset_text_field='text',
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

wandb.finish()