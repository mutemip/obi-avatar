"""
avatar_engine.py
Wraps Wav2Lip inference and converts the output video into
a sequence of QPixmap frames that the Qt UI can display.

Wav2Lip pipeline:
  avatar.png  +  response.wav  ->  [Wav2Lip]  ->  output.mp4  ->  frames
"""
import hashlib
import logging
import os
import subprocess
import sys

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
        self._video_cache: dict[str, str] = {}
        self._scan_cache()
        self._evict_cache()

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

    # ── Wav2Lip generation ────────────────────────────────────────────────

    def generate_talking_video(self, audio_path: str,
                               out_path: str = None,
                               resize_factor: int = 2) -> str:
        if not self.wav2lip_available:
            log.warning("Wav2Lip unavailable — skipping lip-sync generation.")
            return ""

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
                log.error("Wav2Lip stdout:\n%s", result.stdout[-2000:])
                log.error("Wav2Lip stderr:\n%s", result.stderr[-2000:])
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
