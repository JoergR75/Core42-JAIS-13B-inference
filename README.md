# JAIS 13B Inference on AMD ROCm

This Python script provides a simple interface to run **JAIS 13B** language model inference on AMD GPUs using **ROCm** and Hugging Face Transformers. It allows generating text from custom prompts with configurable sampling parameters.

## Features

- Load and run the `core42/jais-13b-chat` model from Hugging Face.
- Supports prompts in any language.
- Configurable `max_new_tokens`, `temperature`, and `top_p` for text generation.
- Optional chat template formatting.
- Automatic device selection between ROCm-enabled GPU (`cuda`) and CPU.

## Requirements

- Python 3.10+
- PyTorch with ROCm support
- Transformers library
- Hugging Face account and API token
- Sufficient GPU memory (~24–32 GB VRAM recommended for 13B model)

## Usage

Run the script with a prompt:

```bash
python3 jais_13b_inference_test.py \
  --model core42/jais-13b-chat \
  --prompt "اكتب فقرة قصيرة عن أهمية البيانات في الرعاية الصحية." \
  --hf_token YOUR_HF_TOKEN

or

```bash
python3 jais_13b_inference_test.py \
  --model core42/jais-13b-chat \
  --prompt "Give me three bullet points on AMD ROCm for AI." \
  --hf_token YOUR_HF_TOKEN
