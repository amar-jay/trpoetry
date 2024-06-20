from contextlib import nullcontext
import torch
import tiktoken
from model import GPT, GPTConfig
import os

block_size = 512
batch_size=4
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
num_samples = 200
max_new_tokens = 500
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions

top_k = 20 # retain only the top_k most likely tokens, clamp others to have 0 probability

ctx = nullcontext()

# torch.manual_seed(1234)

ckpt_path = os.path.join('data', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig()

model = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

model = torch.compile(model) 

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode('\n')
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        with open("generated.txt", "w") as f: 
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated = decode(y[0].tolist())
                print(f"generating... {k} / {num_samples}")
                
                f.write(generated)
                f.write('---------------')

            
print("DONE")

