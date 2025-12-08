# Goal
Benchmarking the performance of llama.cpp on a CPU.

# Variables (status)
- CPU core count: implemented (threads sweep 1/half/all cores; override with `--threads`).
- CPU frequency: not implemented; control externally (e.g., `cpupower frequency-set`) and rerun to compare.
- Model size: implemented via repo list sweep (Gemma 270M/1B/4B/12B).
- Sequence length: implemented (prompt/gen sweeps; override with `--n-prompt`/`--n-gen`).
- Quantization: implemented (Q4_K_M/Q8_0/Q2_K_L sweeps; override with `--quantization`).
- Memory bandwidth: not implemented; simulate via external throttling/cgroup controls and rerun to compare.

# What to measure
- prefill speed
- inference speeds

# Code
- bench.py that uses llama.cpp/ folder, varying all the variables above, run the inference, and save results to result/ folder with results in json + graphs
- venv/ environment - use it

# Models
unsloth/gemma-3-270M-it-GGUF
unsloth/gemma-3-1B-it-GGUF
unsloth/gemma-3-4B-it-GGUF
unsloth/gemma-3-12B-it-GGUF

# Quantizations
- Q4_K_M
- Q8_0
- Q2_K_L

# Summary of tests (bench.py defaults)
- Use the venv: `./venv/bin/python bench.py`
- Defaults sweep the models/quantizations above, download them into `models-cache/`, run llama-bench, and update `result/result.json` after each run.
- If downloads fail, install `huggingface_hub` in the venv. Plots require `matplotlib` (`pip install matplotlib`).
- Models: unsloth Gemma 270M, 1B, 4B, 12B (GGUF) with quantizations Q4_K_M, Q8_0, Q2_K_L (downloaded to `models-cache/`).
- Threads: sweep all logical CPU counts 1..N for OMP/LLAMA/GGML thread envs by default; override via `--threads 1,2,...`.
- Shapes: sweep batch sizes 512/1024/2048, micro-batch 128/256/512, prompt 128/512 tokens, generate 128/512 tokens (change with CLI flags).
- Repetitions: 3 runs per config passed to `llama-bench` (`--repetitions`).
- Bench binary: `llama.cpp/build/bin/llama-bench`; results persist incrementally to `result/result.json` with per-run metadata and raw stdout/stderr. Plots (e.g., throughput vs threads/batch, latency vs threads) are written alongside the JSON in `result/`.
