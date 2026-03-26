# Obi — Observability AI Assistant

A desktop AI assistant with a lip-synced talking avatar that answers questions about your application portfolio using RAG (Retrieval-Augmented Generation) and a local LLM via Ollama.

## Features

- **Streaming responses** — LLM tokens stream to the chat bubble in real time; audio starts playing as soon as the first sentence is ready
- **Sentence-by-sentence TTS + lip-sync** — each sentence is independently synthesized and lip-synced in parallel, reducing perceived latency
- **RAG Knowledge Base** — loads CSV data, embeds with Ollama, and retrieves relevant context for every query
- **Local LLM** — powered by Ollama (no cloud APIs, fully offline capable)
- **Lip-synced Avatar** — Wav2Lip generates talking-head video for each sentence; swaps in mid-playback when ready
- **Voice Input** — speech-to-text via faster-whisper (click the mic button)
- **Text Input** — standard chat interface
- **Neural TTS** — Microsoft Edge neural voices via edge-tts

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| [Ollama](https://ollama.com) | Local LLM + embeddings |
| FFmpeg | Audio/video conversion |

## Setup

### 1. Install Ollama and pull models

```bash
# Install Ollama (see https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM and embedding models
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_build.txt
```

### 3. Prepare assets

- Place your avatar image as `avatar.png` in the project root
- Place the Wav2Lip checkpoint at `Wav2Lip/checkpoints/wav2lip_gan.pth`
  - Download from the [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip)

### 4. Knowledge base

Place your data as a CSV file in the project root. The default path is configured in `config.py`:

```python
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "dummy_observability_app_data_20_records.csv")
```

The CSV should have columns like: `Application ID`, `App Name`, `Application Tier`, `Incident TTR (hrs)`, `Monitoring Level`, `Observability Risk Score`.

## Running

```bash
# Make sure Ollama is running
ollama serve   # if not already running as a service

# Start the app
python main.py
```

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Ollama LLM model name |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge-TTS voice name |
| `ENABLE_LIPSYNC` | `True` | Toggle Wav2Lip lip-sync generation |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size for STT |

## Project Structure

```
├── main.py               # Entry point — PyQt5 UI with chat + avatar
├── config.py             # All configuration settings
├── avatar_engine.py      # Wav2Lip inference + frame extraction
├── voice_engine.py       # TTS, audio playback, mic recording, STT
├── knowledge_base.py     # RAG: CSV loading, Ollama embeddings, vector search
├── llm_engine.py         # Ollama LLM integration
├── requirements_build.txt
├── avatar.png            # Your avatar image
├── dummy_observability_app_data_20_records.csv
└── Wav2Lip/              # Wav2Lip model + inference
    ├── inference.py
    └── checkpoints/
        └── wav2lip_gan.pth
```

## Architecture — Streaming Pipeline

```
User Query (text or voice)
    │
    ├─► [faster-whisper]  STT (if voice)
    │
    ├─► [KnowledgeBase]   Embed query → cosine similarity → top-k docs
    │
    ├─► [Ollama LLM]      Streams tokens to chat bubble in real time
    │         │
    │         ├─► Sentence 1 detected → [edge-tts] → play audio immediately
    │         │                           └─► [Wav2Lip] → swap in lip-sync mid-playback
    │         │
    │         ├─► Sentence 2 detected → [edge-tts] → queue behind sentence 1
    │         │                           └─► [Wav2Lip] → parallel generation
    │         │
    │         └─► Sentence N …
    │
    └─► [PyQt5 UI]        Seamless sentence-by-sentence playback
```

## Changing the LLM or Knowledge Base

- **Different LLM**: Change `OLLAMA_MODEL` in `config.py` (any model available via `ollama list`)
- **Different knowledge base**: Replace the CSV file and update `KNOWLEDGE_BASE_PATH` in `config.py`
- **Different embedding model**: Change `OLLAMA_EMBED_MODEL` in `config.py`
