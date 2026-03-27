"""
Configuration for Obi — Observability AI Assistant
"""
import os
import sys
import shutil

# ─── Frozen-app detection (PyInstaller)
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    BASE_DIR = os.path.dirname(sys.executable)
    _INTERNAL = os.path.join(BASE_DIR, "_internal")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _INTERNAL = BASE_DIR

# ─── Paths ────────
AVATAR_IMAGE        = os.path.join(BASE_DIR, "avatar.png")
WAV2LIP_DIR         = os.path.join(BASE_DIR, "Wav2Lip")
WAV2LIP_CHECKPOINT  = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
TEMP_DIR            = os.path.join(BASE_DIR, "temp")

for _d in (TEMP_DIR,):
    os.makedirs(_d, exist_ok=True)

# ─── Embedded Python (for Wav2Lip subprocess in frozen builds) 
if FROZEN:
    _EMBED_PYTHON = os.path.join(BASE_DIR, "_python", "python.exe")
    if not os.path.isfile(_EMBED_PYTHON):
        _EMBED_PYTHON = None
else:
    _EMBED_PYTHON = None

PYTHON_FOR_SUBPROCESS = _EMBED_PYTHON or sys.executable

# ─── FFmpeg
if FROZEN:
    _ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    FFMPEG_BIN = os.path.join(BASE_DIR, _ffmpeg_name)
    if not os.path.isfile(FFMPEG_BIN):
        FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
else:
    FFMPEG_BIN = shutil.which("ffmpeg")
    if not FFMPEG_BIN:
        try:
            import imageio_ffmpeg
            FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            FFMPEG_BIN = "ffmpeg"

# ─── Ollama (LLM + Embeddings)
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_MODEL       = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ─── Knowledge Base 
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR,
                                   "dummy_observability_app_data_20_records.csv")

# ─── Greeting 
GREETING_TEXT = ("Hello! I'm Obi, your observability assistant. "
                 "I will make you understand your applications, risk scores, "
                 "and monitoring levels.")

# ─── Voice / STT 
TTS_VOICE          = "en-US-AriaNeural"
WHISPER_MODEL_SIZE = "base"

# ─── Feature Toggles 
ENABLE_LIPSYNC = True

# ─── UI 
WINDOW_TITLE     = "Obi — Observability Assistant"
WINDOW_W         = 1100
WINDOW_H         = 720
AVATAR_DISPLAY_W = 360
AVATAR_DISPLAY_H = 360

# Dark theme palette
COLOR_BG        = "#0d0f14"
COLOR_PANEL     = "#13161f"
COLOR_CARD      = "#1a1e2b"
COLOR_ACCENT    = "#4f8ef7"
COLOR_ACCENT2   = "#7c5cbf"
COLOR_TEXT      = "#e8eaf0"
COLOR_SUBTEXT   = "#7a8099"
COLOR_USER_MSG  = "#1e3a5f"
COLOR_BOT_MSG   = "#1a1e2b"
COLOR_BORDER    = "#252a3a"
