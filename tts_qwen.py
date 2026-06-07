"""
Uso do modelo Qwen3-TTS-12Hz-0.6B-CustomVoice (memescreamer).
Idiomas: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
Speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee.
"""
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_ID = "memescreamer/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """Usa CUDA se disponível, senão CPU."""
    # #region agent log
    _cuda = torch.cuda.is_available()
    _dev = "cuda:0" if _cuda else "cpu"
    try:
        _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"get_device","timestamp":' + str(int(__import__("time").time()*1000)) + ',"location":"tts_qwen.get_device","message":"device choice","data":{"cuda_available":' + str(_cuda).lower() + ',"returned_device":"' + _dev + '"},"hypothesisId":"A"}\n'); _f.close()
    except Exception: pass
    # #endregion
    if _cuda:
        return "cuda:0"
    return "cpu"


def get_device_info():
    """Retorna string descritiva do dispositivo em uso (para exibir na UI)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"GPU: {name}"
    return "CPU (sem GPU)"


def load_model(device_map=None, use_flash_attn=True):
    """Carrega o modelo. Flash Attention 2 opcional (requer GPU compatível)."""
    device_map = device_map or get_device()
    kwargs = {
        "device_map": device_map,
        "dtype": torch.bfloat16 if device_map != "cpu" else torch.float32,
    }
    if use_flash_attn and device_map != "cpu":
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        m = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
        # #region agent log
        try:
            _ds = str(getattr(m, "device", None)); _pd = str(next(m.model.parameters()).device) if hasattr(m, "model") else "n/a"
            _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"load_model_done","timestamp":' + str(int(__import__("time").time()*1000)) + ',"location":"tts_qwen.load_model","message":"model loaded","data":{"device_map":"' + str(device_map) + '","model_device":' + repr(_ds) + ',"param_device":' + repr(_pd) + '},"hypothesisId":"B"}\n'); _f.close()
        except Exception: pass
        # #endregion
        return m
    except Exception:
        kwargs.pop("attn_implementation", None)
        m = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
        # #region agent log
        try:
            _ds = str(getattr(m, "device", None)); _pd = str(next(m.model.parameters()).device) if hasattr(m, "model") else "n/a"
            _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"load_model_fallback","timestamp":' + str(int(__import__("time").time()*1000)) + ',"location":"tts_qwen.load_model","message":"model loaded (no flash_attn)","data":{"device_map":"' + str(device_map) + '","model_device":' + repr(_ds) + ',"param_device":' + repr(_pd) + '},"hypothesisId":"B"}\n'); _f.close()
        except Exception: pass
        # #endregion
        return m


def synthesize(
    text: str,
    language: str = "Portuguese",
    speaker: str = "Ryan",
    instruct: str | None = None,
    output_path: str | None = None,
    model=None,
    **gen_kwargs,
):
    """
    Gera áudio a partir do texto.
    instruct: instrução em linguagem natural (ex: "fale com tom muito feliz").
    gen_kwargs: parâmetros opcionais repassados ao modelo (temperature, top_k, top_p, repetition_penalty, do_sample, etc.).
    """
    if model is None:
        model = load_model()
    kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    kwargs.setdefault("do_sample", False)  # greedy = mais rápido que sampling
    kwargs.setdefault("max_new_tokens", 8)   # ≤2s de geração (áudio ~0.5–0.7 s)
    # #region agent log
    _t0 = __import__("time").time()
    _kwargs_log = {k: v for k, v in kwargs.items() if k in ("do_sample", "max_new_tokens")}
    try:
        _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"synthesize_start","timestamp":' + str(int(_t0*1000)) + ',"location":"tts_qwen.synthesize","message":"gen_kwargs","data":' + __import__("json").dumps(_kwargs_log) + ',"hypothesisId":"speed"}\n'); _f.close()
    except Exception: pass
    # #endregion
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct or "",
        **kwargs,
    )
    # #region agent log
    _t1 = __import__("time").time()
    # #endregion
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "qwen_tts_output.wav")
    sf.write(output_path, wavs[0], sr)
    # #region agent log
    _t2 = __import__("time").time()
    try:
        _sg = round(_t1 - _t0, 2); _sw = round(_t2 - _t1, 2); _st = round(_t2 - _t0, 2)
        _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"synthesize_timing","timestamp":' + str(int(_t2*1000)) + ',"location":"tts_qwen.synthesize","message":"timing","data":{"sec_generate":' + str(_sg) + ',"sec_write":' + str(_sw) + ',"sec_total":' + str(_st) + ',"audio_len_sec":' + str(round(len(wavs[0])/sr, 2)) + '},"hypothesisId":"speed"}\n'); _f.close()
    except Exception: pass
    # #endregion
    return output_path, sr


if __name__ == "__main__":
    # Exemplo: gerar um áudio em português
    path, sample_rate = synthesize(
        text="Olá! Este é um exemplo de síntese de voz com Qwen3-TTS.",
        language="Portuguese",
        speaker="Ryan",
        instruct="fale de forma clara e amigável",
    )
    print(f"Áudio salvo em: {path} (sample rate: {sample_rate})")
