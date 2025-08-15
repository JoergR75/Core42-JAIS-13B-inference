# Jais 13B Inference on AMD ROCm

This Python script provides a simple interface to run **JAIS 13B** language model inference on AMD GPUs using **ROCm** and Hugging Face Transformers. It allows generating text from custom prompts with configurable sampling parameters.

## Features

- Load and run the `core42/jais-13b-chat` model from Hugging Face.
- Supports prompts in any language.
- Configurable `max_new_tokens`, `temperature`, and `top_p` for text generation.
- Optional chat template formatting.
- Automatic device selection between ROCm-enabled GPU (`cuda`) and CPU.

## Requirements

- ROCm 6.4.2+ (Link to the automated script: https://github.com/JoergR75/ROCm-6.4.2-deployment-on-RDNA4)
- Python 3.12+
- PyTorch 2.9.0 with ROCm support (2.9.0.dev20250729+rocm6.4)
- Transformers library
- Hugging Face account and API token
- Sufficient GPU memory (~24–32 GB VRAM recommended for 13B model)
- Tested on 2x Radeon RX 9070 (16GB) and INSTINCT MI210 (64GB)

## Usage

Run the script with an Arabic prompt:

```bash
python3 jais_13b_inference_test.py \
  --model core42/jais-13b-chat \
  --prompt "اكتب فقرة قصيرة عن أهمية البيانات في الرعاية الصحية." \
  --hf_token YOUR_HF_TOKEN
```
or with an English prompt

```bash
python3 jais_13b_inference_test.py \
  --model core42/jais-13b-chat \
  --prompt "Give me three bullet points on AMD ROCm for AI." \
  --hf_token YOUR_HF_TOKEN
```
=== Generated Text ===

Give me three bullet points on AMD ROCm for AI.

1.  ROCm is a multi-core architecture designed for AI workloads.
2.  ROCm includes a runtime that supports dynamic parallelism and fine-grained task scheduling.
3.  ROCm includes a compiler that supports auto-tuning and auto-vectorization.
