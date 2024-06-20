import torch; from torch import nn
import numpy as np
from model import GPTConfig, GPT
import math
import os
import time
from contextlib import nullcontext

block_size = 512
batch_size=4
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
learning_rate = 6e-4 # max learning rate
max_iters = 6000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
eval_iters = 100
warmup_iters = 50
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
lr_decay_iters = 6000 # should be ~= max_iters per Chinchilla
decay_lr = True
gradient_accumulation_steps = 5  # used to simulate larger batch sizes

ctx = nullcontext()



torch.manual_seed(1234)

gptconf = GPTConfig()

model = GPT(gptconf)
model.to(device)
# print(model)

if block_size < model.config.block_size:
    print("cropping block_size ...")
    model.crop_block_size(block_size)

#
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device)

# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split in ['train', 'test', 'val']:
        data = np.memmap(os.path.join('data', f'{split}.bin'), dtype=np.uint16, mode='r')
    else:
        raise NameError

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# print(get_batch('test'))
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
running_mfu = -1.0
iter_num = 0

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


logfile = open("log.txt", "w+", encoding='utf-8')
best_val_loss = 1e9
logits, loss = None, None
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % 100 == 0:
        losses = estimate_loss()
        print(f"step {iter_num:5.0f} | train loss {losses['train']:8.4f} | val loss {losses['val']:8.4f}")
        logfile.write(f"iter: {iter_num}, 'train/loss': losses['train'], 'val/loss': losses['val'], 'lr': {lr}, 'mfu': {running_mfu*100}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to data/ckpt.pt")
                torch.save(checkpoint, os.path.join('data', 'ckpt.pt'))


    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        X, Y = get_batch('train')

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    #
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % 10 == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

logfile.close()
