"""
voice_engine.py
Text-to-speech  : edge-tts  (Microsoft neural voices, free)
Speech-to-text  : faster-whisper  (local Whisper via CTranslate2)
Audio playback  : sounddevice / winsound / aplay / ffplay
Mic recording   : sounddevice
"""
import asyncio
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import wave

import numpy as np
import edge_tts

from config import TTS_VOICE, TEMP_DIR, FFMPEG_BIN, WHISPER_MODEL_SIZE

log = logging.getLogger(__name__)
os.makedirs(TEMP_DIR, exist_ok=True)

_IS_WINDOWS = platform.system() == "Windows"

# ── Optional dependency detection 

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except (ImportError, OSError) as e:
    _HAS_SOUNDDEVICE = False
    log.warning("sounddevice unavailable (%s). Mic recording and playback disabled.", e)

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except (ImportError, OSError):
    _HAS_SOUNDFILE = False

_HAS_WHISPER = False
try:
    from faster_whisper import WhisperModel as _WhisperModel
    _HAS_WHISPER = True
except ImportError:
    log.warning("faster-whisper not installed — speech-to-text disabled.")

_APLAY  = shutil.which("aplay")
_PAPLAY = shutil.which("paplay")

if _IS_WINDOWS:
    import winsound


class VoiceEngine:
    def __init__(self):
        self._playback_proc = None

        # Mic recording state
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._stream = None
        self._sample_rate = 16_000

        # Whisper model (lazy-loaded on first transcription)
        self._whisper: _WhisperModel | None = None

        log.info(
            "VoiceEngine ready (aplay=%s, windows=%s, mic=%s, stt=%s).",
            _APLAY is not None, _IS_WINDOWS,
            _HAS_SOUNDDEVICE, _HAS_WHISPER,
        )

    # ── TTS 

    def synthesize(self, text: str, out_path: str | None = None) -> str:
        if out_path is None:
            out_path = os.path.join(TEMP_DIR, "response.wav")

        async def _synth():
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            mp3_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.write(chunk["data"])
            mp3_buf.seek(0)

            mp3_path = out_path.replace(".wav", ".mp3")
            with open(mp3_path, "wb") as f:
                f.write(mp3_buf.read())

            if self._convert_mp3_to_wav(mp3_path, out_path):
                return out_path
            return mp3_path

        loop = asyncio.new_event_loop()
        result_path = loop.run_until_complete(_synth())
        loop.close()
        log.info("TTS written to %s", result_path)
        return result_path

    @staticmethod
    def _convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
        try:
            from pydub import AudioSegment
            AudioSegment.converter = FFMPEG_BIN
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
            return True
        except Exception as e:
            log.debug("pydub conversion failed: %s — trying ffmpeg CLI", e)

        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", mp3_path,
                 "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=30, check=True
            )
            return True
        except Exception as e:
            log.warning("ffmpeg conversion failed: %s", e)
        return False

    # ── Mic recording 

    @property
    def mic_available(self) -> bool:
        return _HAS_SOUNDDEVICE

    @property
    def stt_available(self) -> bool:
        return _HAS_WHISPER

    def start_recording(self):
        if not _HAS_SOUNDDEVICE:
            log.error("Cannot record — sounddevice not available.")
            return
        self._audio_chunks = []
        self._recording = True

        def _callback(indata, _frames, _time, _status):
            if self._recording:
                self._audio_chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            callback=_callback,
        )
        self._stream.start()
        log.info("Mic recording started.")

    def stop_recording(self) -> str:
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio_chunks:
            log.warning("No audio captured.")
            return ""

        audio = np.concatenate(self._audio_chunks, axis=0)
        out_path = os.path.join(TEMP_DIR, "mic_recording.wav")
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio.tobytes())

        log.info("Mic recording saved to %s (%d samples).", out_path, len(audio))
        return out_path

    # ── Speech-to-text 
    def transcribe(self, audio_path: str) -> str:
        if not _HAS_WHISPER:
            log.error("faster-whisper not installed — cannot transcribe.")
            return ""
        if not os.path.isfile(audio_path):
            return ""

        if self._whisper is None:
            log.info("Loading Whisper model '%s' (first use)…", WHISPER_MODEL_SIZE)
            self._whisper = _WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )

        segments, _info = self._whisper.transcribe(audio_path, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        log.info("Transcription: %s", text)
        return text

    # ── Playback 

    def play_audio_nonblocking(self, path: str):
        self.stop_playback()
        abs_path = os.path.abspath(path)

        if _IS_WINDOWS:
            def _win_play():
                try:
                    winsound.PlaySound(
                        abs_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                except Exception as e:
                    log.error("winsound playback failed: %s", e)
            threading.Thread(target=_win_play, daemon=True).start()
            log.info("Audio playback started (winsound): %s", abs_path)
            return

        if _HAS_SOUNDDEVICE and _HAS_SOUNDFILE:
            def _sd_play():
                try:
                    data, sr = sf.read(abs_path, dtype="float32")
                    sd.play(data, sr)
                    sd.wait()
                except Exception as e:
                    log.debug("sounddevice playback failed: %s", e)
                    self._subprocess_play_nonblocking(abs_path)
            threading.Thread(target=_sd_play, daemon=True).start()
            log.info("Audio playback started (sounddevice): %s", abs_path)
            return

        self._subprocess_play_nonblocking(abs_path)

    def _subprocess_play_nonblocking(self, path: str):
        for cmd in self._playback_commands(path):
            try:
                self._playback_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                log.info("Audio playback started (pid=%d): %s",
                         self._playback_proc.pid, " ".join(cmd))
                return
            except (FileNotFoundError, OSError):
                continue
        log.error("No audio player available.")

    def stop_playback(self):
        if self._playback_proc and self._playback_proc.poll() is None:
            self._playback_proc.terminate()
            self._playback_proc = None

    @staticmethod
    def _playback_commands(path: str) -> list:
        cmds = []
        if _APLAY:
            cmds.append([_APLAY, "-q", path])
        if _PAPLAY:
            cmds.append([_PAPLAY, path])
        ffplay = FFMPEG_BIN.replace("ffmpeg", "ffplay") if FFMPEG_BIN else None
        if ffplay and shutil.which(ffplay):
            cmds.append([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        return cmds

    def play_audio_async(self, path: str, on_done=None):
        def _play():
            if _IS_WINDOWS:
                try:
                    winsound.PlaySound(
                        os.path.abspath(path),
                        winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                except Exception:
                    pass
            elif _HAS_SOUNDDEVICE and _HAS_SOUNDFILE:
                try:
                    data, sr = sf.read(path, dtype="float32")
                    sd.play(data, sr)
                    sd.wait()
                except Exception:
                    self._subprocess_play_blocking(path)
            else:
                self._subprocess_play_blocking(path)
            if on_done:
                on_done()
        threading.Thread(target=_play, daemon=True).start()

    def _subprocess_play_blocking(self, path: str):
        for cmd in self._playback_commands(path):
            try:
                subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                continue
