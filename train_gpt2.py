#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = (
        50257  # 50000 bpe merges + 256 bytes token + 1 <|endoftext|> token
    )
    num_layers: int = 12
    num_heads: int = 12
    embedding_dim: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0
        # key, query, value projections for all heads in batch
        self.c_attn = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        # not really a bias, more of the mask, but following the OpenAI/HF name
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(emb_dim, dim=2)
        q = q.view(
            batch_size, seq_len, self.num_heads, emb_dim // self.num_heads
        ).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, emb_dim // num_heads)
        k = k.view(
            batch_size, seq_len, self.num_heads, emb_dim // self.num_heads
        ).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, emb_dim // num_heads)
        v = v.view(
            batch_size, seq_len, self.num_heads, emb_dim // self.num_heads
        ).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, emb_dim // num_heads)

        # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1))) # (batch_size, num_heads, seq_len, seq_len)
        # att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, emb_dim // num_heads) -> (batch_size, num_heads, seq_len, emb_dim // num_heads)
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_dim, config.embedding_dim * 4)
        self.c_proj = nn.Linear(config.embedding_dim * 4, config.embedding_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.embedding_dim),
                wpe=nn.Embedding(config.block_size, config.embedding_dim),
                h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                ln_f=nn.LayerNorm(config.embedding_dim),
            )
        )

        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.transformer["wte"].weight
        # init parameters, iterate over all sub module and apply init function
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.size()
        assert (
            seq_len <= self.config.block_size
        ), "Cannot forward, model block size is exhausted"
        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer["wpe"](pos)  # (seq_len, emb_dim)
        tok_emb = self.transformer["wte"](input_ids)  # (batch_size, seq_len, emb_dim)
        x = tok_emb + pos_emb
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        loss = None
        if targets is not None:
            # cross entropy actually is negative log likelihood -ln(1/50257) = 10.824905, so innitial loss need to similiar to this value
            # 50257 means initial uniform distribution
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        # load weights from huggingface
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained model : %s" % model_type)

        config_args = {
            "gpt2": dict(num_layers=12, num_heads=12, embedding_dim=768),  # 124M
            "gpt2-medium": dict(
                num_layers=24, num_heads=16, embedding_dim=1024
            ),  # 350M
            "gpt2-large": dict(num_layers=36, num_heads=20, embedding_dim=1280),  # 774M
            "gpt2-xl": dict(num_layers=48, num_heads=25, embedding_dim=1600),  # 1550M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024  # maximum sequence length
        config = GPT2Config(**config_args)

        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not parameter

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.bias")
        ]  # ignore these, just a buffer
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys) == len(
            sd_hf_keys
        ), f"mismatched keys: {len(sd_keys)} != {len(sd_hf_keys)}"
        for key in sd_hf_keys:
            if any(key.endswith(k) for k in transposed):
                # sd_hf[key].shape[::-1] means [2, 3] -> [3, 2]
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].T)
            else:
                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the cadidates parameters that require gradients
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim group. Any parameters that is 2D will weight decay, others not
        # i.e. all weights in matmul + embeddings will have weight decay, while biases or layernorm will not
        decay_params = [p for pn, p in params_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in params_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
        )

        # create AdamW optimizer with fusion version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print("use fused adamw: %s" % use_fused)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # +1 because we need to predict the next token， label need right shift
        buffer = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

import torch.distributed as dist

# run the training loop
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# open tf32 mode
torch.set_float32_matmul_precision("high")

total_batch_size = 524288  # 2**19, ~0.5M tokens
B, T = 16, 1024
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(
        "total batch size = %d, grad_accum_steps = %d"
        % (total_batch_size, grad_accum_steps)
    )
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)


model = GPT2(GPT2Config(vocab_size=50304))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(step):
    # linear warmup stage
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # if step > decay_steps, return min_lr
    if step > max_steps:
        return min_lr
    # cosine decay stage
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff start at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=[0.9, 0.95], eps=1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        # data.to(device) must return, but model.to(device) is not necessary
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        # import code; code.interact(local=locals())
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    token_processed = B * T * grad_accum_steps * ddp_world_size
    token_per_sec = token_processed / dt
    if master_process:
        print(
            f"step {step} : loss = {loss_accum.item():.6f}, time = {dt*1000:.2f}ms, norm = {norm:.4f}, lr = {lr:.4e}, token/sec = {token_per_sec:.2f}"
        )

if ddp:
    destroy_process_group()


import sys

sys.exit(0)

num_return_sequences = 5
max_length = 30

# model = GPT2.from_pretrained("gpt2")
model = GPT2(GPT2Config())
model.eval()
model.to(device)


# prefix token
prefix = "Hello, how are you?"

enc = tiktoken.get_encoding("gpt2")
input_ids = enc.encode(prefix)  # T = 8
input_ids = torch.tensor(input_ids, dtype=torch.long)
input_ids = input_ids.unsqueeze(0).repeat(num_return_sequences, 1)  # B = 5, T = 8
x = input_ids.to(device)

# generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, V)
        next_token_logits = logits[:, -1, :]  # (B, V)
        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 50, dim=-1)  # (B, 50), (B, 50)
        # means index of the sampled elements
        ix = torch.multinomial(top_probs, 1)  # (B, 1)
        x_col = torch.gather(top_indices, -1, ix)  # (B, 1)
        x = torch.cat((x, x_col), dim=1)  # (B, T+1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    text = enc.decode(tokens)
    print(f"sample {i+1} : {text}")
