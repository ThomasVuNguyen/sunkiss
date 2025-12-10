# Goal

Benchmark and record the performance of CPU-based pre-training of ML models with differing variables (below) to see how the training time changes with respect to each variable.

# Variables

Some variables can be **varied within the same device**, while others can only be **recorded** and compared **between different devices**.

## Hardware

* **CPU core count:** Vary from 1 to all cores
* **CPU frequency:** Vary from minimum supported to maximum supported in ~0.4 GHz increments
* **Instruction set (AVX2, AVX-512, AMX):** Record only
* **Memory bandwidth:** Record only
* **Cache size / hierarchy:** Record only
* **Socket count / NUMA topology:** Record only
* **Thermal / power mode:** Vary (performance governor, turbo on/off)

## System & Kernel

* OMP threading
* Thread pinning vs no pinning
* Parallelism backend (MKL / oneDNN / OpenBLAS)
* PyTorch inter-op / intra-op threads
* CPU affinity
* Compiler optimizations (GCC flags, Clang flags, LTO)

## ML Framework

* NumPy / PyTorch / JAX
* Kernel fusion on/off
* JIT warmup state
* torch.compile (Inductor) vs eager
* Attention backend: manual vs SDPA fused

## Model & Training

* Model size

* Model architecture

* **Model hyperparameters (all vary within device):**

  * Depth (number of layers): *Recommended range: 2–32*
  * Width (hidden dimension): *Recommended range: 128–2048*
  * Number of attention heads: *Recommended range: 2–32*
  * Head dimension: *Recommended range: 32–128*
  * MLP expansion ratio: *Recommended range: 2–4*
  * Dropout: *0.0–0.2 (optional; usually 0 for speed benchmarks)*
  * Activation function: *GELU, ReLU, SiLU*
  * Attention variant: *full, local (128–512 window), sliding window*

* Sequence length: *Recommended range: 128, 256, 512, 1024, 2048*

* Quantization / precision mode: *FP32, BF16, mixed precision, q8, q4, bitnet*
* GQA/MQA (`kv_heads` ≤ `num_heads`) to reduce KV bandwidth
* Batch size sweep (to probe cache hierarchy)

* Dataset size (only affects measurement duration)

* Model size

* Model architecture

* **Model hyperparameters (all vary within device):**

  * Depth (number of layers)
  * Width (hidden dimension)
  * Number of attention heads
  * Head dimension
  * MLP expansion ratio
  * Dropout (if applicable)
  * Activation function (GELU, ReLU, SiLU)
  * Attention variant (full, local, sliding window)

* Sequence length

* Quantization / precision mode

* Dataset size (only affects measurement duration)

* Model size

* Model architecture

* Sequence length

* Quantization / precision mode

* Dataset size (only affects measurement duration)

# Specification

* Pre-training: TBD
* Fine-tuning: TBD

# Additional CPU/Memory tuning

* Allocator: jemalloc vs default (recorded)
* NUMA awareness (record socket/core binding)

# Assumptions

* Everything runs single-batch
* Only Linux applicable
* OMP threading
* Tensor processing:

  * PyTorch (with MKL & IPEX)
  * JAX
* Nothing (or minimal) else runs on the computer


Output:
- venv/ virtual environment for python
- requirements.txt
- bench.py file that runs all the tests automatically
- output/ is where all the output models are at
- data/ is where all the dataset are at. 
  - Use this dataset: ThomasTheMaker/Arc-Corpus-sample. This contains 1k rows of data - just text.
- result/ is where the benchmark result is:
  - what each variables are & what is the training time

Style: As this is a giant experiment, make sure each experiment is small enough to be run quickly, while still gives us a good idea of how the relationships between variables and training time are.


Convo:

codex resume 019b0625-d2a9-7571-babc-e6324b27b22c --yolo
