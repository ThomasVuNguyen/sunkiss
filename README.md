# sunkiss
CPU scaling experiment for transformers inference, pre-training, and fine-tuning

# Variables
- CPU core count
- CPU frequency
- Model size
- Model architecture
- Dataset size
- Framework - numpy or pytorch
- Sequence length
- Quantization
- Memory bandwidth

# Specification

- Inference: llama.cpp
- Pre-training: TBD
- Fine-tuning: TBD

# Assumptions
- Everything will run single-batch
- Only Linux applicable (no macOS or Windows, yet)
- OMP threading
- Tensor processing
  - Pytorch (with MKL & IPEX)
  - JAX
- Nothing (or minimal) else runs on the computer

# Milestones

Milestone 1: 
- Create a inference_run.py file that runs llama.cpp on a CPU
