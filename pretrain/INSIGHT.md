Consolidated insights from CPU benchmark sweeps

- Quantization: q8 is the clear CPU winner—lowest step times and highest tokens/sec. q4/bitnet (emulated) are slower; fp32/bf16 are similar to each other and trail q8.
- Threads/pinning: Best per-thread efficiency at 1–2 threads with pinning. Unpinned multithreading can massively hurt step time; scaling beyond a few threads yields diminishing returns and falling efficiency.
- Attention backend: Fused SDPA beats manual attention on CPU, especially when paired with q8. This is the most impactful kernel-level knob.
- GQA/MQA: Using kv_heads=1 reduces bandwidth and speeds attention; combined with q8 + SDPA it gives the fastest torch runs.
- Compilation: torch.compile was enabled with fallback; no dramatic speedup vs eager observed—attention backend and quantization matter more.
- Model scaling: Step time grows roughly in proportion to workload/parameter count. Doubling size yields ~2× (or more) slower steps; the ~8× parameter jump (d2/w256/seq128 → d4/w512/seq256) gives ~4–10× slower steps. No exponential blow-up.
- Batch size: Small batches keep timings low; increasing batch improves tokens/sec until cache/bandwidth limits. Most sweeps used batches 1–4.
- Frameworks: Torch outperforms JAX in these CPU tests; JAX curves are flatter and slower here.

Fast-path recipe for CPU:
Torch + q8 + SDPA + kv_heads=1, threads 1–2 pinned, small-ish batches. Expect near-linear step-time growth with model size.
