Insights from result-2 (reduced torch sweep)

- Quantization: q8 clearly led step time across the matrix; q4/bitnet (emulated) were slower than q8 and often close to fp32/bf16. fp32/bf16 were similar to each other.
- Threading/pinning: With pinning on, 1–2 threads are the sweet spot; unpinned runs inflated step times sharply. Efficiency drops as threads increase.
- Model scaling: Jumping from depth=2,width=256,seq=128 to depth=4,width=512,seq=256 increased median step time ~10× (thread=1,batch=1 slice), consistent with roughly linear growth in workload (~8× parameters/work).
- Batch: Small batches (1–4) kept timings low; larger batches were not swept here.
- Takeaway: For CPU torch, use q8, keep threads low and pinned, and expect roughly proportional growth in step time with model size.
