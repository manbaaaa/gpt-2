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


def load_tokens(filename):
    return torch.tensor(np.load(filename), dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"]

        # get the shard files
        data_dir = "edu_fineweb10B"
        shards = os.listdir(data_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_dir, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        # with open("input.txt", "r", encoding="utf-8") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def reset(self):
        # state
        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # +1 because we need to predict the next token， label need right shift
        buffer = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(shards)
            self.tokens = load_tokens(shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


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
train_loader = DataLoaderLite(
    B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)


model = GPT2(GPT2Config(vocab_size=50304))
model.to(device)
use_compile = (
    False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
)
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


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

# create log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
# for clear the log file
with open(log_file, "w", encoding="utf8") as f:
    pass


for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # evaluate stage
    if (step % 250 == 0) or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss = {val_loss_accum.item():.6f}")
            with open(log_file, "a", encoding="utf8") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if (step > 0 and step % 5000 == 0) or last_step:
                    # save checkpoint
                    checkpoint_path = os.path.join(log_dir, f"checkpoint_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # train stage
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        # data.to(device) must return, but model.to(device) is not necessary
        x, y = x.to(device), y.to(device)
        if ddp:
            # this field is also used by the forward pass.
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
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
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
