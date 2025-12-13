Insights from result-1 (early sweep)

- Model scaling: moving from depth=2, width=256, seq=128 to depth=4, width=512, seq=256 raised median step time ~5–6× (thread=1 slice), roughly proportional to the ~8× parameter/work increase—no exponential blow-up.
- Threading/pinning: 1 thread was consistently fastest. Unpinned multithread runs were often much slower; pinning helped but didn’t close the gap to 1 thread.
- Quantization: q8 was slightly faster than fp32/bf16 at 1 thread; q4/bitnet (fake-quant) were slower. Gains are modest here because quant was emulated.
- General note: Results are noisy due to mixed threading configs; the clearest signal is “stay at 1 thread (or pinned 2) and prefer q8 over other quant modes.”
