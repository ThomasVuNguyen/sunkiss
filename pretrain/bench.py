#!/usr/bin/env python3
"""
Quick-and-focused CPU benchmarking harness for transformer decoder pre-training.
Targets both PyTorch (MKL/IPEX) and JAX (XLA) on CPU, sweeping small configs
and logging wall-clock training times plus system metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import psutil
from datasets import load_dataset
from tqdm import tqdm


def ensure_dirs() -> None:
    for d in ["data", "output", "result"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------- data prep ---------------------- #


@dataclass
class DatasetCache:
    path: Path
    seq_len: int
    max_sequences: int


def encode_to_bytes(text: str, seq_len: int) -> List[np.ndarray]:
    """Byte-level encoder (pad=0, bytes are shifted by +1 to reserve pad)."""
    byte_arr = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    # shift by +1 to keep 0 as padding id
    byte_arr = byte_arr.astype(np.int32) + 1
    chunks: List[np.ndarray] = []
    for start in range(0, len(byte_arr), seq_len):
        window = byte_arr[start : start + seq_len]
        if len(window) < seq_len:
            padded = np.zeros(seq_len, dtype=np.int32)
            padded[: len(window)] = window
            window = padded
        chunks.append(window.astype(np.int32))
    return chunks


def prepare_dataset(seq_len: int, max_sequences: int) -> DatasetCache:
    ensure_dirs()
    cache_path = Path(f"data/arc_bytes_seq{seq_len}_n{max_sequences}.npz")
    if cache_path.exists():
        return DatasetCache(cache_path, seq_len, max_sequences)

    print(f"[data] downloading ThomasTheMaker/Arc-Corpus-sample and caching -> {cache_path}")
    ds = load_dataset("ThomasTheMaker/Arc-Corpus-sample", split="train")

    samples: List[np.ndarray] = []
    for row in tqdm(ds, desc="tokenizing", total=len(ds)):
        text = row.get("text", "")
        samples.extend(encode_to_bytes(text, seq_len))
        if len(samples) >= max_sequences:
            break

    if not samples:
        raise RuntimeError("No samples found in dataset.")
    stacked = np.stack(samples[:max_sequences], axis=0)
    np.savez_compressed(cache_path, input_ids=stacked)
    return DatasetCache(cache_path, seq_len, max_sequences)


def load_cached_dataset(cache: DatasetCache) -> np.ndarray:
    arr = np.load(cache.path, allow_pickle=False)["input_ids"]
    return arr


# ---------------------- system helpers ---------------------- #


def set_threading(num_threads: int, interop_threads: Optional[int] = None) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_DYNAMIC"] = "0"
    os.environ["MKL_DYNAMIC"] = "0"
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(interop_threads if interop_threads is not None else num_threads)
    except Exception:
        pass


def maybe_set_cpu_affinity(cores: List[int]) -> None:
    try:
        p = psutil.Process()
        p.cpu_affinity(cores)
    except Exception as exc:  # affinity unsupported / unprivileged
        print(f"[warn] could not set cpu affinity: {exc}")


def collect_sysinfo() -> Dict[str, object]:
    cpu_model = None
    cpu_flags = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        text = cpuinfo_path.read_text(errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
            if line.lower().startswith("flags"):
                cpu_flags = line.split(":", 1)[1].strip()
            if cpu_model and cpu_flags:
                break

    uname = os.uname() if hasattr(os, "uname") else None
    uname_dict = (
        {"sysname": uname.sysname, "nodename": uname.nodename, "release": uname.release, "version": uname.version, "machine": uname.machine}
        if uname
        else None
    )
    return {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "cpu_model": cpu_model,
        "cpu_flags": cpu_flags,
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_bytes": psutil.virtual_memory().total,
        "platform": uname_dict,
        "allocator": {
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "malloc_conf": os.environ.get("MALLOC_CONF", ""),
        },
        "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
    }


# ---------------------- torch path ---------------------- #


def make_torch_model(
    vocab_size: int,
    seq_len: int,
    depth: int,
    width: int,
    num_heads: int,
    kv_heads: int,
    mlp_ratio: float,
    dropout: float,
    attention_variant: str,
    attention_backend: str,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    kv_heads = max(1, min(kv_heads, num_heads))
    if num_heads % kv_heads != 0:
        kv_heads = num_heads

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class SelfAttentionCore(nn.Module):
        def __init__(self, d_model: int, num_heads: int, kv_heads: int, dropout: float, variant: str, window: int, backend: str):
            super().__init__()
            self.num_heads = num_heads
            self.kv_heads = kv_heads
            self.variant = variant
            self.window = window
            self.backend = backend
            head_dim = d_model // num_heads
            self.q_proj = nn.Linear(d_model, num_heads * head_dim)
            self.k_proj = nn.Linear(d_model, kv_heads * head_dim)
            self.v_proj = nn.Linear(d_model, kv_heads * head_dim)
            self.out_proj = nn.Linear(num_heads * head_dim, d_model)
            self.dropout = dropout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq, _ = x.shape
            head_dim = self.q_proj.out_features // self.num_heads
            q = self.q_proj(x).view(bsz, seq, self.num_heads, head_dim).transpose(1, 2)  # (B, H, S, D)
            k = self.k_proj(x).view(bsz, seq, self.kv_heads, head_dim).transpose(1, 2)  # (B, Kv, S, D)
            v = self.v_proj(x).view(bsz, seq, self.kv_heads, head_dim).transpose(1, 2)

            if self.kv_heads != self.num_heads:
                repeat = self.num_heads // self.kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

            idx = torch.arange(seq, device=x.device)
            diff = idx[:, None] - idx[None, :]
            if self.variant == "local":
                mask = torch.where((diff <= 0) & (diff >= -self.window), 0.0, float("-inf"))
            elif self.variant == "sliding":
                mask = torch.where((diff <= 0) & (diff >= -self.window), 0.0, float("-inf"))
            else:
                mask = torch.triu(torch.full((seq, seq), float("-inf"), device=x.device), 1)

            if self.backend == "sdpa":
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=self.dropout, is_causal=False
                )  # (B, H, S, D)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                scores = scores + mask
                attn = torch.softmax(scores, dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                attn_out = torch.matmul(attn, v)

            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq, self.num_heads * head_dim)
            return self.out_proj(attn_out)

    class DecoderBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, kv_heads: int, mlp_ratio: float, dropout: float, attention_variant: str, attention_backend: str):
            super().__init__()
            window = 128 if attention_variant == "local" else 256
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            backend = attention_backend if attention_variant == "full" or attention_backend == "sdpa" else "manual"
            self.attn = SelfAttentionCore(d_model, num_heads, kv_heads, dropout, attention_variant, window, backend)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, int(d_model * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(d_model * mlp_ratio), d_model),
                nn.Dropout(dropout),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.mlp(self.ln2(x))
            return x

    class DecoderModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
            self.pos = PositionalEncoding(width, seq_len)
            self.layers = nn.ModuleList(
                [
                    DecoderBlock(width, num_heads, kv_heads, mlp_ratio, dropout, attention_variant, attention_backend)
                    for _ in range(depth)
                ]
            )
            self.ln = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            h = self.pos(h)
            for blk in self.layers:
                h = blk(h)
            h = self.ln(h)
            logits = self.head(h)
            return logits

    return DecoderModel()


def apply_torch_quantization(model, mode: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if mode == "fp32":
        return model
    if mode in {"bf16", "mixed"}:
        return model  # handled via autocast at runtime

    if mode == "q8":
        try:
            quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            return quantized
        except Exception as exc:
            print(f"[warn] q8 quantization failed ({exc}); using original model")
            return model

    class FakeQuantLinear(nn.Linear):
        def __init__(self, linear: nn.Linear, bits: int):
            super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None)
            self.bits = bits
            with torch.no_grad():
                self.weight.copy_(linear.weight)
                if linear.bias is not None:
                    self.bias.copy_(linear.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.bits == 4:
                qmin, qmax = -8, 7
            else:
                qmin, qmax = -1, 1
            scale = float(self.weight.abs().max() / max(abs(qmin), abs(qmax)) + 1e-8)
            zp = 0
            w_q = torch.fake_quantize_per_tensor_affine(self.weight, scale, zp, qmin, qmax)
            return F.linear(x, w_q, self.bias)

    def convert(module: nn.Module, bits: int) -> nn.Module:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, FakeQuantLinear(child, bits))
            else:
                convert(child, bits)
        return module

    if mode == "q4":
        return convert(model, bits=4)
    if mode == "bitnet":
        return convert(model, bits=1)
    return model


def torch_train_step(model, optimizer, loss_fn, batch, precision: str):
    import torch

    x = batch
    target = x  # simple LM objective on next token shift could be added for realism
    model.train()
    optimizer.zero_grad()
    if precision in {"bf16", "mixed"}:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
    else:
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def run_torch(config: "RunConfig", batches: Iterable[np.ndarray]) -> Dict[str, object]:
    import torch
    from tqdm import tqdm

    set_threading(config.num_threads, interop_threads=config.inter_op_threads)
    if config.pin_threads:
        cores = list(range(config.num_threads))
        maybe_set_cpu_affinity(cores)

    model = make_torch_model(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        depth=config.depth,
        width=config.width,
        num_heads=config.num_heads,
        kv_heads=config.kv_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attention_variant=config.attention_variant,
        attention_backend=config.attention_backend,
    )
    model = apply_torch_quantization(model, config.precision)
    compiled = False
    if config.compilation_mode == "compile":
        try:
            model = torch.compile(model, fullgraph=True, mode="max-autotune")
            compiled = True
        except Exception as exc:
            print(f"[warn] torch.compile failed ({exc}); falling back to eager")
    model = model.to("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    torch.manual_seed(0)
    durations = []
    iterator = tqdm(
        batches,
        total=config.steps,
        desc=f"{config.framework}-{config.precision}-seq{config.seq_len}-thr{config.num_threads}",
        leave=False,
    )
    for step_idx, batch_np in enumerate(iterator):
        if step_idx >= config.steps:
            break
        batch = torch.tensor(batch_np, dtype=torch.long)
        t0 = time.time()
        loss_val = torch_train_step(model, optimizer, loss_fn, batch, precision=config.precision)
        step_dur = time.time() - t0
        durations.append(step_dur)
        iterator.set_postfix({"last_loss": f"{loss_val:.4f}", "step_s": f"{step_dur:.3f}"}, refresh=False)
    return {"loss_last": loss_val, "durations": durations, "compiled": compiled}


# ---------------------- jax path ---------------------- #


def make_jax_model(vocab_size: int, seq_len: int, depth: int, width: int, num_heads: int, kv_heads: int, mlp_ratio: float, dropout: float, attention_variant: str):
    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen import dot_product_attention

    class LocalSelfAttention(nn.Module):
        window: int
        num_heads: int
        kv_heads: int
        head_dim: int

        @nn.compact
        def __call__(self, x, deterministic: bool):
            qkv = nn.Dense(self.num_heads * self.head_dim + self.kv_heads * self.head_dim * 2)(x)
            split1 = self.num_heads * self.head_dim
            split2 = split1 + self.kv_heads * self.head_dim
            q, k, v = jnp.split(qkv, [split1, split2], axis=-1)
            q = q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            k = k.reshape(x.shape[0], x.shape[1], self.kv_heads, self.head_dim)
            v = v.reshape(x.shape[0], x.shape[1], self.kv_heads, self.head_dim)
            if self.kv_heads != self.num_heads:
                repeat = self.num_heads // self.kv_heads
                k = jnp.repeat(k, repeat, axis=2)
                v = jnp.repeat(v, repeat, axis=2)
            seq_len = x.shape[1]
            idx = jnp.arange(seq_len)
            diff = idx[:, None] - idx[None, :]
            attn_mask = jnp.where((diff <= 0) & (diff >= -self.window), 0.0, -1e9)
            out = dot_product_attention(q, k, v, bias=attn_mask, deterministic=deterministic)
            out = out.reshape(x.shape[0], x.shape[1], self.num_heads * self.head_dim)
            out = nn.Dense(self.num_heads * self.head_dim)(out)
            return out

    class DecoderBlock(nn.Module):
        @nn.compact
        def __call__(self, x, deterministic: bool):
            y = nn.LayerNorm()(x)
            if attention_variant == "local":
                attn = LocalSelfAttention(window=128, num_heads=num_heads, kv_heads=kv_heads, head_dim=width // num_heads)(y, deterministic)
            elif attention_variant == "sliding":
                attn = LocalSelfAttention(window=256, num_heads=num_heads, kv_heads=kv_heads, head_dim=width // num_heads)(y, deterministic)
            else:
                attn = nn.SelfAttention(num_heads=num_heads, dtype=jnp.float32)(y)
            x = x + nn.Dropout(dropout)(attn, deterministic=deterministic)

            y = nn.LayerNorm()(x)
            mlp_hidden = int(width * mlp_ratio)
            y = nn.Dense(mlp_hidden)(y)
            y = nn.gelu(y)
            y = nn.Dense(width)(y)
            y = nn.Dropout(dropout)(y, deterministic=deterministic)
            return x + y

    class DecoderModel(nn.Module):
        @nn.compact
        def __call__(self, x, deterministic: bool = False):
            embed = nn.Embed(num_embeddings=vocab_size, features=width, name="embed")(x)
            positions = jnp.arange(seq_len)[None, :]
            pos_emb = nn.Embed(num_embeddings=seq_len, features=width, name="pos_embed")(positions)
            h = embed + pos_emb
            for _ in range(depth):
                h = DecoderBlock()(h, deterministic)
            h = nn.LayerNorm()(h)
            logits = nn.Dense(vocab_size)(h)
            return logits

    return DecoderModel()


def fake_quant_jax(x, bits: int):
    import jax.numpy as jnp

    if bits == 8:
        qmin, qmax = -128, 127
    elif bits == 4:
        qmin, qmax = -8, 7
    else:
        qmin, qmax = -1, 1
    scale = jnp.max(jnp.abs(x)) / max(abs(qmin), abs(qmax) + 1e-6)
    return jnp.clip(jnp.round(x / scale), qmin, qmax) * scale


def run_jax(config: "RunConfig", batches: Iterable[np.ndarray]) -> Dict[str, object]:
    import jax
    import jax.numpy as jnp
    import optax
    from tqdm import tqdm
    from flax.training.train_state import TrainState

    model = make_jax_model(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        depth=config.depth,
        width=config.width,
        num_heads=config.num_heads,
        kv_heads=config.kv_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attention_variant=config.attention_variant,
    )
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, config.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy)

    tx = optax.adamw(3e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def loss_fn(params, batch, rng):
        logits = state.apply_fn(params, batch, rngs={"dropout": rng}, deterministic=False)
        if config.precision in {"bf16", "mixed"}:
            logits = logits.astype(jnp.bfloat16)
        elif config.precision == "q8":
            logits = fake_quant_jax(logits, 8)
        elif config.precision == "q4":
            logits = fake_quant_jax(logits, 4)
        elif config.precision == "bitnet":
            logits = fake_quant_jax(logits, 1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch).mean()
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    durations = []
    loss_val = None
    iterator = tqdm(
        batches,
        total=config.steps,
        desc=f"{config.framework}-{config.precision}-seq{config.seq_len}-thr{config.num_threads}",
        leave=False,
    )
    for step_idx, batch_np in enumerate(iterator):
        if step_idx >= config.steps:
            break
        batch = jnp.array(batch_np)
        t0 = time.time()
        rng, step_rng = jax.random.split(rng)
        loss_val, grads = grad_fn(state.params, batch, step_rng)
        state = state.apply_gradients(grads=grads)
        step_dur = time.time() - t0
        durations.append(step_dur)
        iterator.set_postfix({"last_loss": f"{float(loss_val):.4f}", "step_s": f"{step_dur:.3f}"}, refresh=False)
    return {"loss_last": float(loss_val), "durations": durations}


# ---------------------- runner ---------------------- #


@dataclass
class RunConfig:
    framework: str
    precision: str
    compilation_mode: str
    attention_backend: str
    kv_heads: int
    seq_len: int
    vocab_size: int
    depth: int
    width: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attention_variant: str
    steps: int
    batch_size: int
    num_threads: int
    inter_op_threads: Optional[int]
    pin_threads: bool


def build_single_config(args) -> List[RunConfig]:
    configs: List[RunConfig] = []
    for framework in args.frameworks:
        for precision in args.precisions:
            kv_heads = min(args.kv_heads_options[0] if args.kv_heads_options else args.heads, args.heads)
            configs.append(
                RunConfig(
                    framework=framework,
                    precision=precision,
                    compilation_mode=args.compilation_modes[0] if args.compilation_modes else "eager",
                    attention_backend=args.attention_backends[0] if args.attention_backends else "manual",
                    kv_heads=kv_heads,
                    seq_len=args.seq_len,
                    vocab_size=257,  # byte + pad token
                    depth=args.depth,
                    width=args.width,
                    num_heads=args.heads,
                    mlp_ratio=args.mlp_ratio,
                    dropout=args.dropout,
                    attention_variant=args.attention_variant,
                    steps=args.steps,
                    batch_size=args.batch_options[0] if args.batch_options else args.batch_size,
                    num_threads=args.threads,
                    inter_op_threads=args.inter_threads,
                    pin_threads=args.pin_threads,
                )
            )
    return configs


def build_sweep_configs(args) -> List[RunConfig]:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    if args.thread_options:
        thread_options = sorted({int(t) for t in args.thread_options})
    else:
        thread_options = list(range(1, logical + 1))
    pin_options = [False, True]

    scenarios = [
        {"seq_len": 128, "depth": 2, "width": 256, "heads": 4, "attention_variant": "full"},
        {"seq_len": 256, "depth": 4, "width": 512, "heads": 8, "attention_variant": "local"},
    ]
    configs: List[RunConfig] = []
    for framework in args.frameworks:
        comp_modes = args.compilation_modes if framework == "torch" else ["jit"]
        for precision in args.precisions:
            for compilation_mode in comp_modes:
                for scenario in scenarios:
                    kv_list = []
                    for k in args.kv_heads_options:
                        if k == "num_heads":
                            kv_list.append(scenario["heads"])
                        elif isinstance(k, int) and k <= scenario["heads"]:
                            kv_list.append(k)
                    if not kv_list:
                        kv_list = [scenario["heads"]]
                    for attention_backend in args.attention_backends:
                        backend = attention_backend
                        if scenario["attention_variant"] != "full" and attention_backend == "sdpa":
                            backend = "manual"
                        for kv_heads in kv_list:
                            for batch_size in args.batch_options:
                                for threads in thread_options:
                                    for pin in pin_options:
                                        configs.append(
                                            RunConfig(
                                                framework=framework,
                                                precision=precision,
                                                compilation_mode=compilation_mode,
                                                attention_backend=backend,
                                                kv_heads=kv_heads,
                                                seq_len=scenario["seq_len"],
                                                vocab_size=257,
                                                depth=scenario["depth"],
                                                width=scenario["width"],
                                                num_heads=scenario["heads"],
                                                mlp_ratio=args.mlp_ratio,
                                                dropout=args.dropout,
                                                attention_variant=scenario["attention_variant"],
                                                steps=args.steps,
                                                batch_size=batch_size,
                                                num_threads=threads,
                                                inter_op_threads=args.inter_threads,
                                                pin_threads=pin,
                                            )
                                        )
    return configs


def batched(arr: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(arr), batch_size):
        end = min(len(arr), start + batch_size)
        yield arr[start:end]


def write_result_row(row: Dict[str, object], csv_path: Path, jsonl_path: Path) -> List[str]:
    row_for_csv = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
        except Exception:
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()
    df_new = pd.DataFrame([row_for_csv])
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged.to_csv(csv_path, index=False)

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return list(df_merged.columns)


def config_key(cfg: RunConfig) -> str:
    return "|".join(
        [
            cfg.framework,
            cfg.precision,
            cfg.compilation_mode,
            cfg.attention_backend,
            str(cfg.kv_heads),
            str(cfg.seq_len),
            str(cfg.depth),
            str(cfg.width),
            str(cfg.num_heads),
            f"{cfg.mlp_ratio}",
            f"{cfg.dropout}",
            cfg.attention_variant,
            str(cfg.steps),
            str(cfg.batch_size),
            str(cfg.num_threads),
            str(cfg.inter_op_threads),
            str(cfg.pin_threads),
        ]
    )


def load_completed_configs(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    completed = set()
    for _, row in df.iterrows():
        def get(col, default=None):
            val = row[col] if col in row else default
            if pd.isna(val) and default is not None:
                return default
            return val

        cfg = RunConfig(
            framework=get("framework"),
            precision=get("precision"),
            compilation_mode=get("compilation_mode", "eager"),
            attention_backend=get("attention_backend", "manual"),
            kv_heads=int(get("kv_heads", get("num_heads", 1))),
            seq_len=int(get("seq_len", 0)),
            vocab_size=257,
            depth=int(get("depth", 0)),
            width=int(get("width", 0)),
            num_heads=int(get("num_heads", 1)),
            mlp_ratio=float(get("mlp_ratio", 4.0)),
            dropout=float(get("dropout", 0.0)),
            attention_variant=get("attention_variant", "full"),
            steps=int(get("steps", 10)),
            batch_size=int(get("batch_size", 1)),
            num_threads=int(get("num_threads", 1)),
            inter_op_threads=int(get("inter_op_threads")) if not pd.isna(get("inter_op_threads", None)) else None,
            pin_threads=bool(get("pin_threads", False)),
        )
        completed.add(config_key(cfg))
    return completed


def main():
    parser = argparse.ArgumentParser(description="CPU transformer decoder benchmark (PyTorch + JAX)")
    parser.add_argument("--frameworks", default="torch,jax", help="Comma-separated: torch,jax")
    parser.add_argument("--precisions", default="fp32,bf16,q8,q4,bitnet", help="Comma-separated precision/quant modes")
    parser.add_argument("--mode", choices=["sweep", "single"], default="sweep", help="sweep runs curated matrix by default")
    parser.add_argument("--compilation-modes", default="eager,compile", help="Comma-separated: eager,compile")
    parser.add_argument("--attention-backends", default="manual,sdpa", help="Comma-separated attention backends")
    parser.add_argument("--kv-heads-options", default="1", help="Comma-separated kv_heads options (<= num_heads); use 'num_heads' to match heads")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-variant", choices=["full", "local", "sliding"], default="full")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-options", type=str, default="1,2,4,8,16,32,64", help="Comma-separated batch sizes to sweep")
    parser.add_argument("--threads", type=int, default=max(1, psutil.cpu_count(logical=False) or 1))
    parser.add_argument("--inter-threads", type=int, default=None)
    parser.add_argument("--pin-threads", action="store_true")
    parser.add_argument(
        "--thread-options",
        type=str,
        default="",
        help="Comma-separated thread counts to sweep (defaults to 1..logical cores)",
    )
    parser.add_argument("--max-sequences", type=int, default=256, help="limit cached sequences for quick runs")
    parser.add_argument("--result-path", type=str, default="result/bench_results.csv")
    parser.add_argument("--jsonl", type=str, default="result/bench_results.jsonl")
    args = parser.parse_args()

    args.frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    args.precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    args.thread_options = [t.strip() for t in args.thread_options.split(",") if t.strip()]
    args.compilation_modes = [c.strip() for c in args.compilation_modes.split(",") if c.strip()]
    args.attention_backends = [a.strip() for a in args.attention_backends.split(",") if a.strip()]
    kv_opts = []
    for k in args.kv_heads_options.split(","):
        k = k.strip()
        if not k:
            continue
        if k.lower() == "num_heads":
            kv_opts.append("num_heads")
        else:
            kv_opts.append(int(k))
    args.kv_heads_options = kv_opts
    args.batch_options = [int(b) for b in args.batch_options.split(",") if b.strip()]

    configs = build_sweep_configs(args) if args.mode == "sweep" else build_single_config(args)
    unique_seq_lens = sorted({cfg.seq_len for cfg in configs})

    ensure_dirs()
    data_cache: Dict[int, np.ndarray] = {}
    for sl in unique_seq_lens:
        cache = prepare_dataset(seq_len=sl, max_sequences=args.max_sequences)
        data_cache[sl] = load_cached_dataset(cache)

    sysinfo = collect_sysinfo()
    csv_path = Path(args.result_path)
    jsonl_path = Path(args.jsonl)
    csv_header = None
    completed = load_completed_configs(csv_path)

    print(f"[plan] running {len(configs)} configs in {args.mode} mode")
    overall_start = time.time()
    for idx, cfg in enumerate(configs, 1):
        elapsed = time.time() - overall_start
        if idx > 1:
            avg = elapsed / (idx - 1)
            remaining = avg * (len(configs) - idx + 1)
        else:
            remaining = 0.0
        key = config_key(cfg)
        if key in completed:
            print(
                f"[skip {idx}/{len(configs)} | already in results] {cfg.framework} precision={cfg.precision} "
                f"threads={cfg.num_threads} seq={cfg.seq_len} pin={cfg.pin_threads} batch={cfg.batch_size}"
            )
            continue
        print(
            f"[run {idx}/{len(configs)} | ETA ~{remaining:.1f}s] {cfg.framework} precision={cfg.precision} mode={cfg.compilation_mode} "
            f"attn={cfg.attention_backend} kv={cfg.kv_heads} steps={cfg.steps} batch={cfg.batch_size} "
            f"threads={cfg.num_threads} seq={cfg.seq_len} depth={cfg.depth} width={cfg.width} heads={cfg.num_heads} pin={cfg.pin_threads}"
        )
        print(
            f"    threading env -> OMP={cfg.num_threads}, MKL={cfg.num_threads}, inter-op={cfg.inter_op_threads or cfg.num_threads}, pin={cfg.pin_threads}"
        )
        batches = batched(data_cache[cfg.seq_len], cfg.batch_size)
        if cfg.framework == "torch":
            out = run_torch(cfg, batches)
        elif cfg.framework == "jax":
            out = run_jax(cfg, batches)
        else:
            raise ValueError(f"Unknown framework {cfg.framework}")
        result_row = {
            **asdict(cfg),
            **{"loss_last": out["loss_last"], "durations": out["durations"], "sysinfo": sysinfo},
            "mean_step_time": float(np.mean(out["durations"])),
            "p95_step_time": float(np.percentile(out["durations"], 95)),
        }
        if "compiled" in out:
            result_row["compiled"] = out["compiled"]
        csv_header = write_result_row(result_row, csv_path, jsonl_path)

    print(f"[done] appended results to {csv_path} and {jsonl_path}")


if __name__ == "__main__":
    main()
