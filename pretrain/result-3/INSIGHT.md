Insights from result-3 (q8-focused sweep with compile/SDPA/GQA)

- Quantization: q8 is the clear winner on CPU torch; other precisions cluster much higher step times. Precision bars show q8 at the bottom across configs.
- Threading: Best per-thread efficiency is at 1–2 threads (pinned). Tokens/sec rises slightly then plateaus/declines as threads increase; efficiency falls off sharply with more threads.
- Attention backend: SDPA-backed runs (fused attention) deliver higher throughput than manual attention; gains are most visible in tokens/sec vs threads for the larger model.
- Compilation: torch.compile was enabled; compile failures fall back to eager. No dramatic speedup observed in plots; fused attention + q8 dominate more than compile state.
- GQA (kv=1): Reducing kv_heads to 1 helped bandwidth-bound attention; combined with q8 it yielded the fastest lines in tokens/sec plots.
- Model scaling: Larger model (depth=4,width=512,seq=256) is slower than the small one roughly in proportion to workload (linear-ish growth; no exponential spike).
- Summary: For CPU torch, use q8 + SDPA + kv_heads=1, keep threads low and pinned; expect near-linear step-time growth with model size.
