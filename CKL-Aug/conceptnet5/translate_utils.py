import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100Tokenizer, M2M100ForConditionalGeneration

# ===== 配置：本地翻译模型路径 =====
MT_PATH = "/mnt/3BBFCD1C4844CE10/guowei/opus-mt-zh-en"
MT_PATH_back = "/mnt/3BBFCD1C4844CE10/guowei/m2m100_1.2B"

# ===== 模型只加载一次（非常重要） =====
_mt_tokenizer = AutoTokenizer.from_pretrained(MT_PATH, local_files_only=True)
_mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_PATH,local_files_only=True,device_map="auto",torch_dtype="auto")
_mt_model.eval()

_mt_tokenizer_back = M2M100Tokenizer.from_pretrained(MT_PATH_back, local_files_only=True)
_mt_model_back = M2M100ForConditionalGeneration.from_pretrained(MT_PATH_back, local_files_only=True, torch_dtype="auto")
_mt_model_back.eval()

@torch.no_grad()
def translate_zh_to_en(text: str,num_beams,max_token) -> str:
    """
    中文 -> 英文（opus-mt-zh-en）
    只返回英文字符串
    """
    if not text or not str(text).strip():
        return ""

    inputs = _mt_tokenizer([text],return_tensors="pt",truncation=True).to(_mt_model.device)
    out = _mt_model.generate(**inputs,max_new_tokens=max_token,num_beams=num_beams,do_sample=False)

    en = _mt_tokenizer.decode(out[0],skip_special_tokens=True).strip()

    return en

@torch.no_grad()
def translate_en_to_zh(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    _mt_tokenizer_back.src_lang = "en"
    inputs = _mt_tokenizer_back(text, return_tensors="pt", truncation=True).to(_mt_model_back.device)
    out = _mt_model_back.generate(
        **inputs,
        forced_bos_token_id=_mt_tokenizer_back.get_lang_id("zh"),
        max_new_tokens=6,
        num_beams=1,          # 词汇建议关 beam
        do_sample=False
    )
    return _mt_tokenizer_back.decode(out[0], skip_special_tokens=True).strip()
