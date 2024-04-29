from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.manifold
torch.manual_seed(23)

MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inverse_eps_proj16/checkpoint-27500'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inverse_eps_proj32_beta0.1/checkpoint-27500'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inverse_eps_proj32_beta0.1_new_soft/checkpoint-37500'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inverse_eps_proj32_beta0.1_new_soft_normalize/checkpoint-10000'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/cluster_detach_dropout0.4_router0.001_inverse_eps_proj32_beta0.1_new_soft_normalize2/checkpoint-7500'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/new_proj32_soft/checkpoint-25000'
MODEL_PATH = '/fs/nexus-scratch/vatsalb/mixtral/new_proj16_soft2/checkpoint-40000'
device = 'cuda'

model = MixtralForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')

test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='/fs/nexus-scratch/vatsalb/huggingface')

ATTENTION_DROPOUT = 0.4
CLUSTER_EXPERTS = True
GUMBEL_SOFTMAX = False
DETACH_INPUT = CLUSTER_EXPERTS and False
LEARNING_RATE = 4e-5
ROUTER_COEF = 0.001
DISTANCE_TYPE = 'inverse_eps'
CLUSTER_DIM = 32

# config = MixtralConfig(
#   vocab_size=32000,
#   hidden_size=4096 // 4,
#   intermediate_size=14336 // 4,
#   num_hidden_layers=32 // 4,
#   num_attention_heads=32 // 2,
#   num_key_value_heads=8,
#   hidden_act='silu',
#   max_position_embeddings=4096 * 32 // 32,
#   initializer_range=0.02,
#   rms_norm_eps=1e-5,
#   use_cache=True,
#   pad_token_id=None,
#   bos_token_id=1,
#   eos_token_id=2,
#   tie_word_embeddings=False,
#   rope_theta=1e6,
#   sliding_window=None,
#   attention_dropout=ATTENTION_DROPOUT,
#   num_experts_per_tok=2,
#   num_local_experts=8 // 2,
#   output_router_logits=True, #False,
#   router_aux_loss_coef=ROUTER_COEF,
#   cluster_experts=CLUSTER_EXPERTS,
#   gumbel_softmax=GUMBEL_SOFTMAX,
#   detach_input=DETACH_INPUT,
#   distance_type=DISTANCE_TYPE,
#   cluster_dim=CLUSTER_DIM,
#   cluster_std=2
# )
# model = MixtralForCausalLM(config).to(device)
# model.eval()

text = '''
He moved on in the summer of 759 ; this has traditionally been ascribed to famine , but Hung believes that frustration is a more likely reason . He next spent around six weeks in Qinzhou ( now Tianshui , Gansu province ) , where he wrote more than sixty poems .
'''
text = '\n\n'.join(test.shuffle()['text'][:50])
# text = 'He bought 8 apples and 5 oranges for a total of 13 fruits'
# text = 'George Steiner , literary critic for The New Yorker and The New York Times, had written about the Holocaust in some of his previous books.'
encodings = tokenizer(text, return_tensors='pt').to(device)

# text = 'He wrote more than sixty poems.'

# text=test['text'][100]

experts0 = torch.tensor([])
experts1 = torch.tensor([])
points = torch.tensor([])
centers = torch.tensor([])

def hook(module, input, output):
  global experts0
  global experts1
  global points
  global centers
  # points = module.norm(module.proj(input[0])).clone().detach().cpu()
  points = input[0] @ module.proj
  points = points.clone().detach().cpu()
  print('hi')
  print(input[0].mean(dim=0))
  print(input[0].mean())
  print(points.mean(dim=0))
  print(points.mean())
  print(torch.norm(input[0], dim=1).mean())
  print(torch.norm(input[0].cpu() @ torch.nn.functional.normalize(torch.randn((1024, 32))/32, dim=1), dim=1).mean())
  print(torch.norm(input[0] @ module.proj, dim=1).mean())
  # print(input[0].mean(0))
  # print(points.mean(0))
  print(points.shape)
  centers = module.centers.T.clone().detach().cpu() # module.proj(module.centers.T).clone().detach().cpu()
  U, S, V = torch.pca_lowrank(torch.cat([points, centers], dim=0))
  proj_points = torch.matmul(points, V[:, :2]).detach().cpu()
  proj_centers = torch.matmul(centers, V[:, :2]).detach().cpu()
  # X_tsne = sklearn.manifold.TSNE().fit_transform(torch.cat([points, centers], dim=0).numpy())
  # proj_points = X_tsne[:points.size(0)]
  # proj_centers = X_tsne[points.size(0):]
  # print(proj_points.shape)
  # print(proj_centers.shape)
  fig, ax = plt.subplots(figsize=(20, 20))
  ax.scatter(proj_points[:, 0], proj_points[:, 1])
  ax.scatter(proj_centers[:, 0], proj_centers[:, 1], s=100)
  for i in range(points.size(0)):
    ax.annotate(tokenizer.decode(encodings.input_ids[0, i]), (proj_points[i, 0], proj_points[i, 1]), fontsize=10)
  for i in range(centers.size(0)):
    ax.annotate(i, (proj_centers[i, 0], proj_centers[i, 1]), fontsize=10)
  fig.savefig('cluster.png')
  plt.close(fig)
  print(output)
  print(torch.nn.functional.softmax(output, dim=1))
  print(torch.nn.functional.softmax(output, dim=1).argsort(dim=1))
  experts0 = torch.nn.functional.softmax(output, dim=1).argsort(dim=1)[:, 0]
  experts1 = torch.nn.functional.softmax(output, dim=1).argsort(dim=1)[:, 1]

gate_hook = model.model.layers[0].block_sparse_moe.gate.register_forward_hook(hook)

outputs = model(encodings.input_ids)

gate_hook.remove()

print([((experts0 == i).float().mean()+(experts1 == i).float().mean())/2 for i in range(4)])

print([(experts0 == i).float().mean() for i in range(4)])
print([(experts1 == i).float().mean() for i in range(4)])






'''how often are nearest neighbors assigned to same expert'''
dists = (points.pow(2).sum(1, keepdim=True) - 2 * points @ points.T + points.T.pow(2).sum(0, keepdim=True))
nn = dists.argsort(dim=0)[1]
print((experts0 == experts0[nn]).float().mean())



'''coloring in expert assigments'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_colored_text(words, categories, colors):
    fig, ax = plt.subplots(figsize=(len(encodings.input_ids[0]), 4))
    ax.set_axis_off()
    # Calculate the x position for each word
    x_positions = []
    x = 0
    for word in words:
        x_positions.append(x)
        x += len(word) + 1  # Add 1 for space
    # Plot each word with its corresponding color
    for word, category, x in zip(words, categories, x_positions):
        ax.text(x, 0, word, color='black', fontsize=16, ha='left')
        ax.add_patch(Rectangle((x - 0.2, -0.2), len(word), 0.4, color=colors[category], alpha=0.3))
    ax.set_xlim(0, x_positions[-1])
    ax.set_ylim(0, 1)
    plt.savefig('test.png')

words = [tokenizer.decode(id) for id in encodings.input_ids[0]]
for i, word in enumerate(words):
  if word == '':
    words[i] = ' '

colors = ['red', 'blue', 'green', 'orange']

plot_colored_text(words, experts0.tolist(), colors)
'''
proj 16 beta:
0.1 [tensor(0.0106, device='cuda:0'), tensor(0.4959, device='cuda:0'), tensor(0.0149, device='cuda:0'), tensor(0.4787, device='cuda:0')]
0.2 [tensor(0.3031, device='cuda:0'), tensor(0.0234, device='cuda:0'), tensor(0.2636, device='cuda:0'), tensor(0.4099, device='cuda:0')]
0.4 [tensor(0.3348, device='cuda:0'), tensor(0.0389, device='cuda:0'), tensor(0.3970, device='cuda:0'), tensor(0.2293, device='cuda:0')]
'''