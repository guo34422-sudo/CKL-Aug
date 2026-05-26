import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration
)


def load_zh_en_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto"
    )

    model.eval()

    return tokenizer, model


def load_en_zh_model(model_path):
    tokenizer = M2M100Tokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = M2M100ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto"
    )

    model.eval()

    return tokenizer, model


@torch.no_grad()
def translate_zh_to_en(text, tokenizer, model, num_beams=4, max_tokens=64):
    text = (text or "").strip()

    if not text:
        return ""

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=False
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


@torch.no_grad()
def translate_en_to_zh(text, tokenizer, model, num_beams=1, max_tokens=6):
    text = (text or "").strip()

    if not text:
        return ""

    tokenizer.src_lang = "en"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("zh"),
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=False
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh_en_model_path", type=str, required=True)
    parser.add_argument("--en_zh_model_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    zh_en_tokenizer, zh_en_model = load_zh_en_model(args.zh_en_model_path)
    en_zh_tokenizer, en_zh_model = load_en_zh_model(args.en_zh_model_path)

    text_en = translate_zh_to_en(
        args.text,
        zh_en_tokenizer,
        zh_en_model
    )

    text_zh = translate_en_to_zh(
        text_en,
        en_zh_tokenizer,
        en_zh_model
    )

    print(text_en)
    print(text_zh)


if __name__ == "__main__":
    main()
