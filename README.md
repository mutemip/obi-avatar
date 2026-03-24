# MS Proactive

MS Proactive is a Python-based project for audio-visual speech synthesis and lip-syncing, leveraging the Wav2Lip model. It provides tools for generating realistic talking head videos by syncing lip movements to speech audio.

## Features
- Audio-visual speech synthesis using Wav2Lip
- Face detection and preprocessing utilities
- Evaluation scripts for generated videos
- Configurable and extensible architecture

## Project Structure
- `main.py` - Main entry point for running the application
- `avatar_engine.py`, `voice_engine.py` - Core logic for avatar and voice processing
- `config.py` - Configuration settings
- `Wav2Lip/` - Contains the Wav2Lip model, training, inference, and evaluation scripts
- `face_detection/` - Face detection models and utilities
- `requirements.txt` - Python dependencies for Wav2Lip
- `requirements_build.txt` - Build dependencies

## Getting Started
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements_build.txt
   ```
3. **Download pre-trained models:**
   - Place the `wav2lip_gan.pth` file in `Wav2Lip/checkpoints/` (already included)
4. **Run the main application:**
   ```bash
   python main.py
   ```

## Usage
- Use the scripts in `Wav2Lip/` for training, inference, and evaluation.
- Refer to the `README.md` files in subfolders for more details on specific modules.


