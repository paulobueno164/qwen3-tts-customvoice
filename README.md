# 🗣️ Qwen3-TTS CustomVoice

> Síntese de voz (text-to-speech) **multilíngue e controlável por instrução** usando o modelo
> [Qwen3-TTS-12Hz CustomVoice](https://huggingface.co/memescreamer/Qwen3-TTS-12Hz-0.6B-CustomVoice),
> com interface web e API Python.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

**[Português](#-português) · [English](#-english)**

---

## 🇧🇷 Português

### Sobre

Projeto de **engenharia de modelos de voz**: empacota um modelo TTS aberto do Hugging Face em uma
aplicação utilizável — página web para digitar texto, gerar, ouvir e baixar o áudio — e expõe uma
API Python limpa para reuso. Suporta **10 idiomas**, **9 vozes** e **controle por instrução em
linguagem natural** ("fale com tom muito feliz").

### Destaques técnicos

- **Carregamento e inferência de modelo PyTorch** com cache do modelo entre gerações (carrega uma
  vez, reutiliza) para reduzir latência.
- **Aceleração por GPU (CUDA)** com fallback automático para CPU; instruções de setup para
  PyTorch CUDA 12.4 e *flash-attention* opcional.
- **Dois modos de uso**: interface web (Gradio) e script/módulo Python.
- **API ergonômica**: `load_model()` + `synthesize(text, language, speaker, instruct)`.

### Stack

Python · PyTorch · Hugging Face Transformers · Gradio · CUDA (opcional).

### Como rodar

```bash
pip install -r requirements.txt
python app.py          # interface web em http://localhost:7860
python tts_qwen.py     # script direto → outputs/qwen_tts_output.wav
```

Como módulo:

```python
from tts_qwen import load_model, synthesize

model = load_model()                       # carrega uma vez
path, sr = synthesize(
    text="Seu texto aqui.",
    language="Portuguese",
    speaker="Ryan",
    instruct="fale de forma clara",
    model=model,
)
```

**Parâmetros** — `language`: Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, Italian · `speaker`: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden,
Ono_Anna, Sohee · `instruct`: instrução livre de estilo/tom.

---

## 🇺🇸 English

### About

A **voice-model engineering** project: it wraps an open TTS model from Hugging Face into a usable
application — a web page to type text, generate, listen and download audio — and exposes a clean
Python API for reuse. Supports **10 languages**, **9 voices** and **natural-language style control**
("speak in a very happy tone").

### Technical highlights

- **PyTorch model loading and inference** with model caching across generations (load once, reuse)
  to cut latency.
- **GPU (CUDA) acceleration** with automatic CPU fallback; setup notes for PyTorch CUDA 12.4 and
  optional flash-attention.
- **Two usage modes**: a Gradio web UI and a Python script/module.
- **Ergonomic API**: `load_model()` + `synthesize(text, language, speaker, instruct)`.

### Stack

Python · PyTorch · Hugging Face Transformers · Gradio · CUDA (optional).

### Getting started

```bash
pip install -r requirements.txt
python app.py          # web UI at http://localhost:7860
python tts_qwen.py     # direct script → outputs/qwen_tts_output.wav
```

---

<sub>Autor / Author: **Paulo Bueno** · [github.com/paulobueno164](https://github.com/paulobueno164)</sub>
