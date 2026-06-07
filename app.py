"""
Página simples: digitar texto → processar → áudio para play, pausar e baixar.
"""
import os
import gradio as gr
import uuid
from tts_qwen import load_model, synthesize, OUTPUT_DIR, get_device_info

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _gen_kwargs(temperature, top_k, top_p, repetition_penalty):
    """Monta kwargs de geração; None = usar padrão do modelo."""
    d = {}
    if temperature is not None and temperature > 0:
        d["temperature"] = float(temperature)
    if top_k is not None and top_k > 0:
        d["top_k"] = int(top_k)
    if top_p is not None and 0 < top_p <= 1:
        d["top_p"] = float(top_p)
    if repetition_penalty is not None and repetition_penalty > 0:
        d["repetition_penalty"] = float(repetition_penalty)
    return d


def gerar_audio(texto, idioma, speaker, instrucao, temperature, top_k, top_p, repetition_penalty):
    if not texto or not texto.strip():
        return None, "Digite um texto para gerar o áudio."
    try:
        model = get_model()
        # #region agent log
        try:
            import torch as _torch
            _f = open(r"c:\projetos\voice\.cursor\debug.log", "a", encoding="utf-8"); _f.write('{"id":"gerar_audio_model","timestamp":' + str(int(__import__("time").time()*1000)) + ',"location":"app.gerar_audio","message":"model at generate","data":{"cuda_available":' + str(_torch.cuda.is_available()).lower() + ',"model_device":' + repr(str(getattr(model, "device", None))) + '},"hypothesisId":"D,E"}\n'); _f.close()
        except Exception: pass
        # #endregion
        path = os.path.join(OUTPUT_DIR, f"tts_{uuid.uuid4().hex[:8]}.wav")
        synthesize(
            text=texto.strip(),
            language=idioma,
            speaker=speaker,
            instruct=(instrucao or "").strip() or None,
            output_path=path,
            model=model,
            **_gen_kwargs(temperature, top_k, top_p, repetition_penalty),
        )
        return path, "Áudio gerado. Use o player abaixo para ouvir ou baixar."
    except Exception as e:
        return None, f"Erro ao gerar áudio: {e}"


IDIOMAS = [
    "Portuguese",
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Chinese",
    "Japanese",
    "Korean",
    "Russian",
]
SPEAKERS = [
    "Ryan",
    "Aiden",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
]

with gr.Blocks(title="Texto para Voz — Qwen3-TTS", theme=gr.themes.Soft()) as app:
    gr.Markdown(f"## Texto para Voz\n**{get_device_info()}** — Digite o texto, ajuste idioma e voz (opcional) e clique em **Gerar áudio**.")
    with gr.Row():
        texto = gr.Textbox(
            label="Texto",
            placeholder="Ex: Olá! Este é um exemplo de síntese de voz.",
            lines=4,
            max_lines=8,
        )
    with gr.Row():
        idioma = gr.Dropdown(label="Idioma", choices=IDIOMAS, value="Portuguese")
        speaker = gr.Dropdown(label="Voz", choices=SPEAKERS, value="Ryan")
    instrucao = gr.Textbox(
        label="Instrução de estilo / entonação (opcional)",
        placeholder="Ex: fale com tom calmo e claro; ou: speak in a very happy tone",
        lines=1,
    )
    with gr.Accordion("Parâmetros avançados (opcional)", open=False):
        gr.Markdown("Ajuste a variabilidade da geração. Deixe em 0 para usar o padrão do modelo.")
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature (maior = mais variado)")
        top_k = gr.Number(value=0, minimum=0, maximum=200, step=1, label="Top-k (0 = padrão)")
        top_p = gr.Number(value=0, minimum=0, maximum=1, step=0.05, label="Top-p (0 = padrão)")
        repetition_penalty = gr.Number(value=0, minimum=0, maximum=2, step=0.1, label="Repetition penalty (0 = padrão)")
    btn = gr.Button("Gerar áudio", variant="primary")
    status = gr.Markdown("")
    audio_out = gr.Audio(label="Áudio gerado", type="filepath", interactive=False)

    def run(texto, idioma, speaker, instrucao, temperature, top_k, top_p, repetition_penalty):
        path, msg = gerar_audio(texto, idioma, speaker, instrucao, temperature, top_k, top_p, repetition_penalty)
        return path, msg

    btn.click(
        fn=run,
        inputs=[texto, idioma, speaker, instrucao, temperature, top_k, top_p, repetition_penalty],
        outputs=[audio_out, status],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
