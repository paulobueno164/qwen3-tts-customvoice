# Qwen3-TTS CustomVoice

Uso do modelo [memescreamer/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/memescreamer/Qwen3-TTS-12Hz-0.6B-CustomVoice) para síntese de voz.

## Setup

```powershell
cd c:\projetos\voice
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Usar GPU (NVIDIA, ex.: RTX 3050)

Por padrão o `pip install -r requirements.txt` instala PyTorch para CPU. Para usar a GPU:

1. Desinstale torch e torchaudio e instale a versão com CUDA (Python 3.13, CUDA 12.4):
   ```powershell
   pip uninstall torch torchaudio -y
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
2. Verifique: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`  
   Deve mostrar `True` e o nome da placa.
3. Na página web, no topo aparecerá **GPU: NVIDIA GeForce ...** quando a GPU estiver em uso.

(Opcional) Para inferência ainda mais rápida em GPU: `pip install flash-attn` (pode exigir compilação no Windows).  
(Opcional) SoX no PATH para algumas operações de áudio: [SoX](http://sox.sourceforge.net/)

## Uso

**Página web (digitar texto → gerar áudio → play / pausar / baixar):**
```powershell
python app.py
```
Abre em `http://localhost:7860`. Na primeira geração o modelo é carregado (pode demorar); depois as gerações ficam mais rápidas.

**Script direto (exemplo em português):**
```powershell
python tts_qwen.py
```
O áudio é salvo em `outputs/qwen_tts_output.wav`.

**Como módulo:**
```python
from tts_qwen import load_model, synthesize

# Carregar uma vez e reutilizar
model = load_model()
path, sr = synthesize(
    text="Seu texto aqui.",
    language="Portuguese",
    speaker="Ryan",
    instruct="fale de forma clara",
    model=model,
)
```

## Parâmetros

- **language:** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **speaker:** Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- **instruct:** Instrução em linguagem natural (ex.: "fale com tom muito feliz", "speak in a very happy tone")

## Estrutura

- `app.py` – página web (Gradio) para digitar texto, gerar e ouvir/baixar o áudio
- `tts_qwen.py` – carrega o modelo e gera áudio
- `outputs/` – arquivos WAV gerados
- `.venv/` – ambiente virtual Python
