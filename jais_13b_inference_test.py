#!/usr/bin/env python3
"""
JAIS inference on AMD ROCm (Hugging Face)

Usage:
  python3 jais_13b_inference_test.py \
    --model core42/jais-13b-chat \
    --prompt "اكتب فقرة قصيرة عن أهمية البيانات في الرعاية الصحية." \
    --hf_token hf_xxxx

or

  python3 jais_13b_inference_test.py \
    --hf_token hf_xxxx \
    --model core42/jais-13b-chat \
    --prompt "Give me three bullet points on AMD ROCm for AI."

"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="JAIS model test script on ROCm")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model", type=str, required=True, help="Model repo name, e.g., core42/jais-13b-chat")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate text for")
    parser.add_argument("--no_chat_template", action="store_true", help="Skip chat template formatting")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_auth_token=args.hf_token,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
        trust_remote_code=True
    )

    # Use pipeline to avoid past_key_values crash
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
#        device=0 if device=="cuda" else -1
    )

    prompt_text = args.prompt
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(args.prompt)
        except Exception:
            pass  # fallback to raw prompt

    # Generate
    outputs = text_gen(
        prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        use_cache=False
    )

    print("\n=== Generated Text ===\n")
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
