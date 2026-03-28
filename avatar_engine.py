"""
avatar_engine.py
Wraps Wav2Lip inference and converts the output video into
a sequence of QPixmap frames that the Qt UI can display.

Wav2Lip pipeline:
  avatar.png  +  response.wav  ->  [Wav2Lip]  ->  output.mp4  ->  frames

Fallback pipeline (no Wav2Lip / no checkpoint):
  avatar.png  +  response.wav  ->  [amplitude-driven jaw warp]  ->  output.mp4
"""
import hashlib
import logging
import os
import subprocess
import sys
import wave as wave_mod

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont
from PyQt5.QtCore import Qt

from config import (AVATAR_IMAGE, WAV2LIP_DIR, WAV2LIP_CHECKPOINT,
                    TEMP_DIR, AVATAR_DISPLAY_W, AVATAR_DISPLAY_H,
                    FFMPEG_BIN, PYTHON_FOR_SUBPROCESS, FROZEN)

log = logging.getLogger(__name__)
os.makedirs(TEMP_DIR, exist_ok=True)

CACHE_DIR = os.path.join(TEMP_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_CACHE_VIDEOS = 50


class AvatarEngine:
    def __init__(self):
        self.wav2lip_available = self._check_wav2lip()
        self.static_frame = self._load_static_avatar()
        self._mouth_y_ratio = self._detect_mouth_region()
        self._video_cache: dict[str, str] = {}
        self._scan_cache()
        self._evict_cache()

    @property
    def can_generate_video(self) -> bool:
        """True if any video generation path is available (Wav2Lip or fallback)."""
        return self.wav2lip_available or (
            os.path.isfile(AVATAR_IMAGE) and bool(FFMPEG_BIN))

    def _check_wav2lip(self) -> bool:
        inference = os.path.join(WAV2LIP_DIR, "inference.py")
        ckpt = WAV2LIP_CHECKPOINT
        if not os.path.isfile(inference):
            log.warning("Wav2Lip inference.py not found at %s.", WAV2LIP_DIR)
            return False
        if not os.path.isfile(ckpt):
            log.warning("Wav2Lip checkpoint not found at %s.", ckpt)
            return False
        if FROZEN and PYTHON_FOR_SUBPROCESS == sys.executable:
            log.warning("Frozen build but no embedded Python found — "
                        "Wav2Lip subprocess will not work.")
            return False
        log.info("Wav2Lip is available.")
        return True

    def _detect_mouth_region(self) -> float:
        """Detect vertical mouth position as a ratio of image height."""
        if not os.path.isfile(AVATAR_IMAGE):
            return 0.72
        img = cv2.imread(AVATAR_IMAGE)
        if img is None:
            return 0.72
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=4)
            if len(faces) > 0:
                _, fy, _, fh = max(faces, key=lambda f: f[2] * f[3])
                return min((fy + int(fh * 0.75)) / img.shape[0], 0.85)
        except Exception:
            pass
        return 0.72

    def _load_static_avatar(self) -> QPixmap:
        if os.path.isfile(AVATAR_IMAGE):
            img = cv2.imread(AVATAR_IMAGE)
            if img is not None:
                return self._cv2_to_pixmap(img)
            log.warning("Could not decode %s — using placeholder.", AVATAR_IMAGE)
        else:
            log.warning("avatar.png not found — using placeholder.")
        return self._create_placeholder()

    @staticmethod
    def _create_placeholder() -> QPixmap:
        pix = QPixmap(AVATAR_DISPLAY_W, AVATAR_DISPLAY_H)
        pix.fill(QColor("#1a1e2b"))
        p = QPainter(pix)
        p.setPen(QColor("#4f8ef7"))
        f = QFont("Segoe UI", 18, QFont.Bold)
        p.setFont(f)
        p.drawText(pix.rect(), Qt.AlignCenter, "Obi\n(place avatar.png here)")
        p.end()
        return pix

    def _wav2lip_env(self) -> dict:
        """Build an env dict that puts ffmpeg on PATH for Wav2Lip subprocess."""
        env = os.environ.copy()
        ffmpeg_dir = self._ensure_ffmpeg_symlink()
        if FROZEN:
            from config import BASE_DIR
            env["PATH"] = (BASE_DIR + os.pathsep +
                           ffmpeg_dir + os.pathsep +
                           env.get("PATH", ""))
        else:
            venv_bin = os.path.dirname(sys.executable)
            env["PATH"] = (venv_bin + os.pathsep +
                           ffmpeg_dir + os.pathsep +
                           env.get("PATH", ""))
        return env

    @staticmethod
    def _ensure_ffmpeg_symlink() -> str:
        """If the ffmpeg binary has a non-standard name (e.g. imageio-ffmpeg),
        create a 'ffmpeg' symlink in TEMP_DIR so shell commands can find it."""
        if not FFMPEG_BIN or not os.path.isfile(FFMPEG_BIN):
            return ""
        if os.path.basename(FFMPEG_BIN) == "ffmpeg":
            return os.path.dirname(FFMPEG_BIN)

        link = os.path.join(TEMP_DIR, "ffmpeg")
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(FFMPEG_BIN, link)
        except OSError as exc:
            log.warning("Could not create ffmpeg symlink: %s", exc)
            return os.path.dirname(FFMPEG_BIN)
        return TEMP_DIR

    # ── Video cache ────────────────────────────────────────────────────────

    def _scan_cache(self):
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith(".mp4"):
                key = fname[:-4]
                self._video_cache[key] = os.path.join(CACHE_DIR, fname)
        if self._video_cache:
            log.info("Loaded %d cached lip-sync videos.", len(self._video_cache))

    def _evict_cache(self):
        """Remove oldest cached videos when exceeding MAX_CACHE_VIDEOS."""
        if len(self._video_cache) <= MAX_CACHE_VIDEOS:
            return
        by_mtime = sorted(
            self._video_cache.items(),
            key=lambda kv: os.path.getmtime(kv[1]) if os.path.isfile(kv[1]) else 0,
        )
        to_remove = len(self._video_cache) - MAX_CACHE_VIDEOS
        for key, path in by_mtime[:to_remove]:
            try:
                os.remove(path)
            except OSError:
                pass
            del self._video_cache[key]
        log.info("Evicted %d cached videos (cap=%d).",
                 to_remove, MAX_CACHE_VIDEOS)

    @staticmethod
    def _audio_hash(audio_path: str) -> str:
        h = hashlib.md5(usedforsecurity=False)
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_cached_video(self, audio_path: str) -> str | None:
        key = self._audio_hash(audio_path)
        path = self._video_cache.get(key)
        if path and os.path.isfile(path):
            log.info("Cache hit for lip-sync video: %s", path)
            return path
        return None

    def generate_talking_video_cached(self, audio_path: str,
                                      resize_factor: int = 2) -> str:
        """Generate lip-sync video, checking cache first and storing result."""
        cached = self.get_cached_video(audio_path)
        if cached:
            return cached

        key = self._audio_hash(audio_path)
        out_path = os.path.join(CACHE_DIR, f"{key}.mp4")
        result = self.generate_talking_video(
            audio_path, out_path=out_path, resize_factor=resize_factor)
        if result:
            self._video_cache[key] = result
            self._evict_cache()
        return result

    # ── Audio extraction from video ───────────────────────────────────────

    def extract_audio_from_video(self, video_path: str,
                                 out_path: str = None) -> str:
        """Extract audio track from an .mp4 video to a .wav file."""
        if not video_path or not os.path.isfile(video_path):
            return ""
        if out_path is None:
            base = os.path.splitext(video_path)[0]
            out_path = base + "_audio.wav"
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", video_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 out_path],
                capture_output=True, timeout=30, check=True,
            )
            if os.path.isfile(out_path):
                log.info("Extracted audio from video: %s", out_path)
                return out_path
        except Exception as e:
            log.warning("Audio extraction from video failed: %s", e)
        return ""

    # ── Fallback animation (amplitude-driven jaw warp) ──────────────────

    def generate_fallback_talking_video(self, audio_path: str,
                                        out_path: str = None,
                                        fps: int = 25) -> str:
        """Lightweight talking animation driven by audio amplitude.
        Works on any platform — no Wav2Lip, no CUDA, no checkpoint needed.
        Uses cv2 remap to warp the lower face proportional to volume."""
        if not os.path.isfile(AVATAR_IMAGE):
            return ""
        img = cv2.imread(AVATAR_IMAGE)
        if img is None or not audio_path or not os.path.isfile(audio_path):
            return ""

        try:
            with wave_mod.open(audio_path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
        except Exception as exc:
            log.warning("Fallback: cannot read audio — %s", exc)
            return ""

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if dtype == np.uint8:
            samples -= 128
        if n_ch > 1:
            samples = samples[::n_ch]

        duration = len(samples) / rate
        n_frames = max(1, int(duration * fps))
        spf = len(samples) / n_frames

        amps = np.array([
            np.sqrt(np.mean(
                samples[int(i * spf):int((i + 1) * spf)] ** 2))
            for i in range(n_frames)
        ])
        peak = amps.max() or 1.0
        amps /= peak

        alpha = 0.35
        for i in range(1, len(amps)):
            amps[i] = alpha * amps[i] + (1 - alpha) * amps[i - 1]

        h, w = img.shape[:2]
        mouth_y = int(h * self._mouth_y_ratio)
        max_shift = max(2, int(h * 0.015))
        below = max(h - mouth_y, 1)

        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        base_map_y = np.tile(
            np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        ys_below = np.arange(mouth_y, h, dtype=np.float32)
        progress = (ys_below - mouth_y) / below

        if out_path is None:
            out_path = os.path.join(TEMP_DIR, "talking_fallback.mp4")
        temp_avi = out_path + ".tmp.avi"

        writer = cv2.VideoWriter(
            temp_avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        for amp in amps:
            if amp < 0.04:
                writer.write(img)
                continue
            shift = float(amp * max_shift)
            map_y = base_map_y.copy()
            map_y[mouth_y:, :] = (
                ys_below - shift * (1 - progress)).reshape(-1, 1)
            frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
            writer.write(frame)

        writer.release()

        try:
            subprocess.run(
                [FFMPEG_BIN, "-y",
                 "-i", temp_avi, "-i", audio_path,
                 "-c:v", "libx264", "-preset", "fast",
                 "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", out_path],
                capture_output=True, timeout=120, check=True,
            )
        except Exception as exc:
            log.warning("Fallback video mux failed: %s", exc)
            return ""
        finally:
            try:
                os.remove(temp_avi)
            except OSError:
                pass

        if os.path.isfile(out_path):
            log.info("Fallback talking video ready: %s", out_path)
            return out_path
        return ""

    # ── Wav2Lip generation ────────────────────────────────────────────────

    def generate_talking_video(self, audio_path: str,
                               out_path: str = None,
                               resize_factor: int = 2) -> str:
        if not self.wav2lip_available:
            log.info("Wav2Lip unavailable — using amplitude-based fallback.")
            return self.generate_fallback_talking_video(audio_path, out_path)

        audio_path = os.path.abspath(audio_path)
        if out_path is None:
            out_path = os.path.join(TEMP_DIR, "talking.mp4")
        out_path = os.path.abspath(out_path)

        cmd = [
            PYTHON_FOR_SUBPROCESS,
            os.path.join(WAV2LIP_DIR, "inference.py"),
            "--checkpoint_path", os.path.abspath(WAV2LIP_CHECKPOINT),
            "--face", os.path.abspath(AVATAR_IMAGE),
            "--audio", audio_path,
            "--outfile", out_path,
            "--pads", "0", "20", "0", "0",
            "--nosmooth",
        ]
        if resize_factor > 1:
            cmd.extend(["--resize_factor", str(resize_factor)])

        def _low_priority():
            try:
                os.nice(10)
            except OSError:
                pass

        log.info("Running Wav2Lip: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=WAV2LIP_DIR, timeout=300,
                env=self._wav2lip_env(),
                preexec_fn=_low_priority,
            )
            if result.returncode != 0:
                log.error("Wav2Lip failed (exit code %d).", result.returncode)
                log.error("Wav2Lip stdout:\n%s", result.stdout[-2000:])
                log.error("Wav2Lip stderr:\n%s", result.stderr[-2000:])
                print(f"\n{'='*60}")
                print("WAV2LIP FAILED — subprocess output below:")
                print(f"{'='*60}")
                print("STDOUT:", result.stdout[-3000:])
                print("STDERR:", result.stderr[-3000:])
                print(f"{'='*60}\n")
                if resize_factor < 2 and "OutOfMemoryError" in result.stderr:
                    log.info("Retrying with --resize_factor 2 …")
                    return self.generate_talking_video(
                        audio_path, out_path, resize_factor=2)
                return ""
            if os.path.isfile(out_path):
                log.info("Talking video ready: %s", out_path)
                return out_path
        except subprocess.TimeoutExpired:
            log.error("Wav2Lip timed out after 300s.")
        except Exception as e:
            log.error("Wav2Lip error: %s", e)
        return ""

    def extract_frames(self, video_path: str) -> list:
        """Extract frames as QPixmaps. MAIN THREAD ONLY."""
        return self.qimages_to_pixmaps(
            self.extract_frames_as_qimages(video_path))

    def extract_frames_as_qimages(self, video_path: str) -> list:
        """Extract frames as QImages (thread-safe)."""
        if not video_path or not os.path.isfile(video_path):
            return []
        images = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            images.append(self._cv2_to_qimage(frame))
        cap.release()
        log.info("Extracted %d frames from %s", len(images), video_path)
        return images

    def qimages_to_pixmaps(self, images: list) -> list:
        """Convert QImages to scaled QPixmaps. MAIN THREAD ONLY."""
        return [self._qimage_to_pixmap(img) for img in images]

    def get_video_fps(self, video_path: str) -> float:
        if not video_path or not os.path.isfile(video_path):
            return 25.0
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    @staticmethod
    def _cv2_to_qimage(frame: np.ndarray) -> QImage:
        """Convert cv2 BGR frame to QImage (thread-safe)."""
        if frame is None:
            return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line,
                      QImage.Format_RGB888).copy()

    @staticmethod
    def _qimage_to_pixmap(qimg: QImage) -> QPixmap:
        """Convert QImage to scaled QPixmap. MAIN THREAD ONLY."""
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(
            AVATAR_DISPLAY_W, AVATAR_DISPLAY_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    @staticmethod
    def _cv2_to_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert cv2 frame to QPixmap. MAIN THREAD ONLY."""
        if frame is None:
            return QPixmap()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line,
                      QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(
            AVATAR_DISPLAY_W, AVATAR_DISPLAY_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def get_idle_pixmap(self) -> QPixmap:
        return self.static_frame

    def get_thinking_frames(self) -> list:
        if not os.path.isfile(AVATAR_IMAGE):
            return [self.static_frame]
        img = cv2.imread(AVATAR_IMAGE)
        if img is None:
            return [self.static_frame]
        img = img.astype(np.float32)
        frames = []
        for alpha in [1.0, 1.08, 1.15, 1.08, 1.0]:
            bright = np.clip(img * alpha, 0, 255).astype(np.uint8)
            frames.append(self._cv2_to_pixmap(bright))
        return frames
