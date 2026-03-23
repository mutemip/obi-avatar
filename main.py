"""
main.py — Obi — Observability AI Assistant — Desktop Application
Entry point.  Run with:  python main.py
"""
import logging
import os
import sys
import threading

from PyQt5.QtCore    import (Qt, QTimer, pyqtSignal, QObject)
from PyQt5.QtGui     import (QColor, QPalette)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QPushButton, QLabel,
                              QGraphicsDropShadowEffect, QStatusBar)

from config        import *
from config        import GREETING_TEXT
from avatar_engine import AvatarEngine
from voice_engine  import VoiceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger(__name__)

GREETING_AUDIO_CACHE = os.path.join(TEMP_DIR, "greeting_cached.wav")
GREETING_VIDEO_CACHE = os.path.join(TEMP_DIR, "greeting_cached.mp4")


# ─── Worker signal bus ────────────────────────────────────────────────────────
class Signals(QObject):
    avatar_frames  = pyqtSignal(list, float, str)
    status         = pyqtSignal(str)
    play_audio     = pyqtSignal(str)
    greeting_ready = pyqtSignal()


# ─── Main Window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals  = Signals()
        self.av_eng   = AvatarEngine()
        self.voice    = VoiceEngine()

        self._is_speaking      = False
        self._frames           = []
        self._frame_idx        = 0
        self._greeting_frames  = []
        self._greeting_fps     = 25.0
        self._greeting_audio   = ""

        self._setup_ui()
        self._connect_signals()

        QTimer.singleShot(500, self._prepare_greeting)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setFixedSize(AVATAR_DISPLAY_W + 60, AVATAR_DISPLAY_H + 200)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }}
            QStatusBar {{
                background: {COLOR_PANEL}; color: {COLOR_SUBTEXT}; font-size: 11px;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 20)
        layout.setSpacing(14)

        name_label = QLabel("Obi")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet(f"""
            color: {COLOR_TEXT}; font-size: 24px;
            font-weight: 700; letter-spacing: 1px;
        """)

        role_label = QLabel("Observability Assistant")
        role_label.setAlignment(Qt.AlignCenter)
        role_label.setStyleSheet(f"color: {COLOR_SUBTEXT}; font-size: 13px;")

        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(AVATAR_DISPLAY_W, AVATAR_DISPLAY_H)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setStyleSheet(f"""
            border: 2px solid {COLOR_BORDER}; border-radius: 16px;
            background: {COLOR_CARD};
        """)
        self._avatar_glow = QGraphicsDropShadowEffect()
        self._avatar_glow.setBlurRadius(30)
        self._avatar_glow.setColor(QColor(COLOR_ACCENT))
        self._avatar_glow.setOffset(0, 0)
        self.avatar_label.setGraphicsEffect(self._avatar_glow)
        self.avatar_label.setPixmap(self.av_eng.get_idle_pixmap())

        self.status_dot = QLabel("● Initializing …")
        self.status_dot.setAlignment(Qt.AlignCenter)
        self.status_dot.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 13px;")

        self.replay_btn = QPushButton("▶  Replay Greeting")
        self.replay_btn.setFixedHeight(40)
        self.replay_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_CARD}; color: {COLOR_ACCENT};
                border: none; border-radius: 10px;
                font-size: 14px; font-weight: 600;
            }}
            QPushButton:hover  {{ background: {COLOR_ACCENT}; color: #fff; }}
            QPushButton:pressed {{ opacity: 0.8; }}
        """)
        self.replay_btn.clicked.connect(self._replay_greeting)
        self.replay_btn.setEnabled(False)

        layout.addWidget(name_label)
        layout.addWidget(role_label)
        layout.addSpacing(6)
        layout.addWidget(self.avatar_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.status_dot)
        layout.addWidget(self.replay_btn)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ── Signals ────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.signals.avatar_frames.connect(self._start_playback)
        self.signals.status.connect(self._status)
        self.signals.play_audio.connect(
            lambda p: self.voice.play_audio_nonblocking(p))
        self.signals.greeting_ready.connect(self._on_greeting_ready)

        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._next_frame)

        self._think_frames = self.av_eng.get_thinking_frames()
        self._think_idx    = 0
        self.think_timer   = QTimer()
        self.think_timer.timeout.connect(self._thinking_frame)

    # ── Greeting ───────────────────────────────────────────────────────────────

    def _prepare_greeting(self):
        self._start_thinking_animation()

        cached = (os.path.isfile(GREETING_VIDEO_CACHE)
                  and os.path.isfile(GREETING_AUDIO_CACHE))
        if cached:
            self._status("Loading greeting …")
        else:
            self._status("Generating greeting (first time, ~30s) …")

        def _worker():
            try:
                if cached:
                    audio_path = GREETING_AUDIO_CACHE
                    video_path = GREETING_VIDEO_CACHE
                else:
                    audio_path = self.voice.synthesize(
                        GREETING_TEXT, out_path=GREETING_AUDIO_CACHE)
                    video_path = self.av_eng.generate_talking_video(
                        audio_path, out_path=GREETING_VIDEO_CACHE)

                if video_path and os.path.isfile(video_path):
                    self._greeting_frames = self.av_eng.extract_frames(video_path)
                    self._greeting_fps = self.av_eng.get_video_fps(video_path)
                    self._greeting_audio = audio_path
                    self.signals.greeting_ready.emit()
                else:
                    self.signals.play_audio.emit(audio_path)
                    self.signals.status.emit("Ready")
            except Exception as e:
                log.error("Greeting failed: %s", e)
                self.signals.status.emit("Ready")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_greeting_ready(self):
        if self._greeting_frames:
            self._start_playback(
                self._greeting_frames,
                self._greeting_fps,
                self._greeting_audio
            )
            self.replay_btn.setEnabled(True)

    def _replay_greeting(self):
        if self._is_speaking:
            return
        if self._greeting_frames and self._greeting_audio:
            self._start_playback(
                self._greeting_frames,
                self._greeting_fps,
                self._greeting_audio
            )

    # ── Avatar frame playback ─────────────────────────────────────────────────

    def _start_thinking_animation(self):
        self._think_idx = 0
        self.think_timer.start(150)

    def _thinking_frame(self):
        if self._think_frames:
            pix = self._think_frames[self._think_idx % len(self._think_frames)]
            self.avatar_label.setPixmap(pix)
            self._think_idx += 1

    def _set_speaking_style(self, speaking: bool):
        if speaking:
            self.avatar_label.setStyleSheet(f"""
                border: 3px solid #27ae60; border-radius: 16px;
                background: {COLOR_CARD};
            """)
            self._avatar_glow.setColor(QColor("#27ae60"))
            self._avatar_glow.setBlurRadius(50)
        else:
            self.avatar_label.setStyleSheet(f"""
                border: 2px solid {COLOR_BORDER}; border-radius: 16px;
                background: {COLOR_CARD};
            """)
            self._avatar_glow.setColor(QColor(COLOR_ACCENT))
            self._avatar_glow.setBlurRadius(30)

    def _start_playback(self, frames: list, fps: float, audio_path: str):
        if not frames:
            return
        self._is_speaking = True
        self.think_timer.stop()
        self._frames    = frames
        self._frame_idx = 0
        interval = max(1, int(1000 / fps))

        self._set_speaking_style(True)
        self._set_dot("● Speaking", "#27ae60")
        self._status("Speaking …")

        if audio_path and os.path.isfile(audio_path):
            self.voice.play_audio_nonblocking(audio_path)

        self.frame_timer.start(interval)

    def _next_frame(self):
        if self._frame_idx >= len(self._frames):
            self.frame_timer.stop()
            self._is_speaking = False
            self.avatar_label.setPixmap(self.av_eng.get_idle_pixmap())
            self._set_speaking_style(False)
            self._set_dot("● Idle", COLOR_SUBTEXT)
            self._status("Ready")
            return
        self.avatar_label.setPixmap(self._frames[self._frame_idx])
        self._frame_idx += 1

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.status_bar.showMessage(msg)
        dot_map = {
            "speak":    ("● Speaking", "#27ae60"),
            "ready":    ("● Idle", COLOR_SUBTEXT),
            "generat":  ("● Generating …", "#e67e22"),
            "loading":  ("● Loading …", "#e67e22"),
            "initial":  ("● Initializing …", COLOR_ACCENT),
        }
        key = msg.lower()
        for k, (label, color) in dot_map.items():
            if k in key:
                self._set_dot(label, color)
                return

    def _set_dot(self, label: str, color: str):
        self.status_dot.setText(label)
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 13px;")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(COLOR_BG))
    palette.setColor(QPalette.WindowText,      QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base,            QColor(COLOR_PANEL))
    palette.setColor(QPalette.AlternateBase,   QColor(COLOR_CARD))
    palette.setColor(QPalette.ToolTipBase,     QColor(COLOR_CARD))
    palette.setColor(QPalette.ToolTipText,     QColor(COLOR_TEXT))
    palette.setColor(QPalette.Text,            QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button,          QColor(COLOR_CARD))
    palette.setColor(QPalette.ButtonText,      QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight,       QColor(COLOR_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
