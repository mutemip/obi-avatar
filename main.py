"""
main.py — Obi — Observability AI Assistant — Desktop Application
Two-panel layout: avatar (left) + chat with RAG (right).
Whole-response lip-sync: LLM response is collected in full, then a
single Wav2Lip video is generated and played with its own audio track.
A "Give me a Second" waiting clip plays while the response generates.
Run with:  python main.py
"""
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QImage, QPalette, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QScrollArea, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect, QStatusBar,
)

from config import *
from avatar_engine import AvatarEngine
from voice_engine import VoiceEngine
from knowledge_base import KnowledgeBase
from llm_engine import LLMEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

GREETING_AUDIO_CACHE = os.path.join(TEMP_DIR, "greeting_cached.wav")
GREETING_VIDEO_CACHE = os.path.join(TEMP_DIR, "greeting_cached.mp4")
WAITING_AUDIO_CACHE  = os.path.join(TEMP_DIR, "waiting_cached.wav")
WAITING_VIDEO_CACHE  = os.path.join(TEMP_DIR, "waiting_cached.mp4")


# ── Qt signals 

class Signals(QObject):
    avatar_frames      = pyqtSignal(list, float, str)
    status             = pyqtSignal(str)
    play_audio         = pyqtSignal(str)
    speak_audio_only   = pyqtSignal(str)
    greeting_ready     = pyqtSignal()
    waiting_ready      = pyqtSignal()
    bot_message        = pyqtSignal(str)
    system_message     = pyqtSignal(str)
    enable_input       = pyqtSignal()
    transcription_done = pyqtSignal(str)

    bot_message_append       = pyqtSignal(str)
    response_video_ready     = pyqtSignal()


# ── Chat bubble widget 

class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self._body_label = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        sender = QLabel("You" if is_user else "Obi")
        sender.setStyleSheet(
            f"color: {'#7a8099' if is_user else COLOR_ACCENT}; "
            f"font-size: 11px; font-weight: bold; background: transparent;"
        )
        timestamp = QLabel(datetime.now().strftime("%H:%M"))
        timestamp.setStyleSheet(
            f"color: {COLOR_SUBTEXT}; font-size: 10px; background: transparent;"
        )

        if is_user:
            header.addStretch()
            header.addWidget(timestamp)
            header.addWidget(sender)
        else:
            header.addWidget(sender)
            header.addWidget(timestamp)
            header.addStretch()

        layout.addLayout(header)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._body_label = body

        bg = COLOR_USER_MSG if is_user else COLOR_BOT_MSG
        body.setStyleSheet(
            f"background: {bg}; color: {COLOR_TEXT}; "
            f"padding: 10px 14px; border-radius: 12px; font-size: 13px;"
        )
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        if is_user:
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(body)
            layout.addLayout(row)
        else:
            row = QHBoxLayout()
            row.addWidget(body)
            row.addStretch()
            layout.addLayout(row)

    def append_text(self, token: str):
        """Append a token to the bubble text (for streaming)."""
        if self._body_label:
            self._body_label.setText(self._body_label.text() + token)


class SystemMessage(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"color: {COLOR_SUBTEXT}; font-size: 11px; "
            f"padding: 6px 12px; background: transparent;"
        )


def _clean_for_tts(text: str) -> str:
    """Strip markdown formatting so TTS reads only words."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Main Window 

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.signals = Signals()

        # Engines
        self.av_eng = AvatarEngine()
        self.voice  = VoiceEngine()
        self.kb     = KnowledgeBase()
        self.llm    = LLMEngine(kb_summary=self.kb.get_summary())

        # Playback state
        self._is_speaking       = False
        self._is_recording      = False
        self._is_processing     = False
        self._frames            = []
        self._frame_idx         = 0

        # Audio-only playback tracking (for mid-stream lip-sync switch)
        self._audio_only_playing = False
        self._audio_start_time   = 0.0
        self._audio_duration_ms  = 0

        # Query generation ID (to cancel stale background lip-sync)
        self._query_id = 0

        # Greeting cache
        self._greeting_frames = []
        self._greeting_fps    = 25.0
        self._greeting_audio  = ""

        # Waiting message cache ("Give me a Second to review this")
        self._waiting_frames = []
        self._waiting_fps    = 25.0
        self._waiting_audio  = ""
        self._waiting_video_playing = False

        # Last response cache (for replay)
        self._last_response_audio  = ""
        self._last_response_frames = []
        self._last_response_fps    = 25.0
        self._last_response_video  = ""

        # Pending response video (built in background thread)
        self._pending_response_frames = []
        self._pending_response_fps    = 25.0
        self._pending_response_audio  = ""
        self._pending_response_video  = ""
        self._current_bot_bubble      = None

        self._setup_ui()
        self._connect_signals()
        self._show_startup_info()

        QTimer.singleShot(500, self._prepare_greeting)

    # ── UI construction 

    def _setup_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_W, WINDOW_H)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                font-family: 'Inter', 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', sans-serif;
            }}
            QStatusBar {{
                background: {COLOR_PANEL}; color: {COLOR_SUBTEXT}; font-size: 11px;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_avatar_panel())
        root.addWidget(self._build_chat_panel(), stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_avatar_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(AVATAR_DISPLAY_W + 60)
        panel.setStyleSheet(f"background: {COLOR_PANEL};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(10)

        name_lbl = QLabel("Obi")
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 22px; font-weight: 700; "
            f"letter-spacing: 1px; background: transparent;"
        )

        role_lbl = QLabel("Observability Assistant")
        role_lbl.setAlignment(Qt.AlignCenter)
        role_lbl.setStyleSheet(
            f"color: {COLOR_SUBTEXT}; font-size: 12px; background: transparent;"
        )

        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(AVATAR_DISPLAY_W, AVATAR_DISPLAY_H)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setStyleSheet(
            f"border: 2px solid {COLOR_BORDER}; border-radius: 14px; "
            f"background: {COLOR_CARD};"
        )
        self._avatar_glow = QGraphicsDropShadowEffect()
        self._avatar_glow.setBlurRadius(30)
        self._avatar_glow.setColor(QColor(COLOR_ACCENT))
        self._avatar_glow.setOffset(0, 0)
        self.avatar_label.setGraphicsEffect(self._avatar_glow)
        self.avatar_label.setPixmap(self.av_eng.get_idle_pixmap())

        self.status_dot = QLabel("● Initializing …")
        self.status_dot.setAlignment(Qt.AlignCenter)
        self.status_dot.setStyleSheet(
            f"color: {COLOR_ACCENT}; font-size: 12px; background: transparent;"
        )

        self.replay_btn = QPushButton("▶  Replay Greeting")
        self.replay_btn.setFixedHeight(36)
        self.replay_btn.setCursor(Qt.PointingHandCursor)
        self.replay_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_CARD}; color: {COLOR_ACCENT};
                border: 1px solid {COLOR_BORDER}; border-radius: 8px;
                font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {COLOR_ACCENT}; color: #fff; }}
        """)
        self.replay_btn.clicked.connect(self._replay)
        self.replay_btn.setEnabled(False)

        layout.addWidget(name_lbl)
        layout.addWidget(role_lbl)
        layout.addSpacing(4)
        layout.addWidget(self.avatar_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.status_dot)
        layout.addWidget(self.replay_btn)
        layout.addStretch()
        return panel

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        header = QLabel("Chat")
        header.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 16px; font-weight: 700; "
            f"background: transparent;"
        )
        layout.addWidget(header)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {COLOR_BG}; border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
            }}
            QScrollBar:vertical {{
                width: 6px; background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {COLOR_BORDER}; border-radius: 3px; min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet(f"background: {COLOR_BG};")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(4)
        self.chat_scroll.setWidget(self.chat_container)

        layout.addWidget(self.chat_scroll, stretch=1)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Ask Obi about your applications…")
        self.text_input.setFixedHeight(40)
        self.text_input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR_CARD}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; border-radius: 10px;
                padding: 0 14px; font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {COLOR_ACCENT};
            }}
        """)
        self.text_input.returnPressed.connect(self._on_send)

        btn_style = f"""
            QPushButton {{
                background: {COLOR_CARD}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; border-radius: 10px;
                font-size: 16px;
            }}
            QPushButton:hover {{ background: {COLOR_ACCENT}; color: #fff; }}
            QPushButton:disabled {{ color: {COLOR_SUBTEXT}; background: {COLOR_PANEL}; }}
        """

        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(40, 40)
        self.mic_btn.setCursor(Qt.PointingHandCursor)
        self.mic_btn.setStyleSheet(btn_style)
        self.mic_btn.setToolTip("Click to start/stop voice recording")
        self.mic_btn.clicked.connect(self._toggle_recording)
        if not self.voice.mic_available:
            self.mic_btn.setEnabled(False)
            self.mic_btn.setToolTip("Install sounddevice for voice input")

        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setStyleSheet(btn_style)
        self.send_btn.setToolTip("Send message")
        self.send_btn.clicked.connect(self._on_send)

        input_row.addWidget(self.text_input, stretch=1)
        input_row.addWidget(self.mic_btn)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        return panel

    # ── Signal wiring 

    def _connect_signals(self):
        self.signals.avatar_frames.connect(self._start_playback)
        self.signals.status.connect(self._update_status)
        self.signals.play_audio.connect(
            lambda p: self.voice.play_audio_nonblocking(p))
        self.signals.speak_audio_only.connect(self._play_audio_only)
        self.signals.greeting_ready.connect(self._on_greeting_ready)
        self.signals.waiting_ready.connect(self._on_waiting_ready)
        self.signals.bot_message.connect(
            lambda t: self._add_chat_bubble(t, is_user=False))
        self.signals.system_message.connect(self._add_system_message)
        self.signals.enable_input.connect(self._enable_input)
        self.signals.transcription_done.connect(self._on_transcription)

        self.signals.bot_message_append.connect(self._append_to_bot_bubble)
        self.signals.response_video_ready.connect(
            self._on_response_video_ready)

        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._next_frame)

        self._think_frames = self.av_eng.get_thinking_frames()
        self._think_idx = 0
        self.think_timer = QTimer()
        self.think_timer.timeout.connect(self._thinking_frame)

        self._audio_done_timer = QTimer()
        self._audio_done_timer.setSingleShot(True)
        self._audio_done_timer.timeout.connect(self._on_audio_only_done)

    # ── Startup info 

    def _show_startup_info(self):
        kb_count = self.kb.doc_count()
        if kb_count:
            self._add_system_message(
                f"Knowledge base loaded: {kb_count} applications")
        else:
            self._add_system_message("⚠ No knowledge base data found")

        if self.llm.is_available():
            self._add_system_message(
                f"Connected to Ollama (model: {OLLAMA_MODEL})")
        else:
            self._add_system_message(
                "⚠ Ollama not reachable — start Ollama and restart the app")

    # ── Greeting 

    def _prepare_greeting(self):
        self._start_thinking_animation()

        cached = (os.path.isfile(GREETING_VIDEO_CACHE)
                  and os.path.isfile(GREETING_AUDIO_CACHE))
        if cached:
            self._update_status("Loading greeting …")
        else:
            self._update_status("Generating greeting (first time, ~30s) …")

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
                    self._greeting_frames = \
                        self.av_eng.extract_frames_as_qimages(video_path)
                    self._greeting_fps = self.av_eng.get_video_fps(video_path)
                    self._greeting_audio = audio_path
                    self.signals.greeting_ready.emit()
                else:
                    if audio_path and os.path.isfile(audio_path):
                        self.signals.speak_audio_only.emit(audio_path)
                    else:
                        self.signals.status.emit("Ready")
            except Exception as e:
                log.error("Greeting failed: %s", e)
                self.signals.status.emit("Ready")

            self.signals.bot_message.emit(GREETING_TEXT)
            self._prepare_waiting_message_bg()

        threading.Thread(target=_worker, daemon=True).start()

    def _prepare_waiting_message_bg(self):
        """Pre-generate the waiting message video (runs in background thread)."""
        cached = (os.path.isfile(WAITING_VIDEO_CACHE)
                  and os.path.isfile(WAITING_AUDIO_CACHE))
        try:
            if cached:
                audio_path = WAITING_AUDIO_CACHE
                video_path = WAITING_VIDEO_CACHE
            else:
                log.info("Generating waiting message video (first time) …")
                audio_path = self.voice.synthesize(
                    WAITING_TEXT, out_path=WAITING_AUDIO_CACHE)
                video_path = self.av_eng.generate_talking_video(
                    audio_path, out_path=WAITING_VIDEO_CACHE)

            if video_path and os.path.isfile(video_path):
                self._waiting_frames = \
                    self.av_eng.extract_frames_as_qimages(video_path)
                self._waiting_fps = self.av_eng.get_video_fps(video_path)
                video_audio = video_path.replace(".mp4", "_audio.wav")
                extracted = self.av_eng.extract_audio_from_video(
                    video_path, video_audio)
                self._waiting_audio = extracted or audio_path
                self.signals.waiting_ready.emit()
            else:
                log.warning("Waiting message video generation failed.")
        except Exception as e:
            log.error("Waiting message preparation failed: %s", e)

    def _on_waiting_ready(self):
        if self._waiting_frames:
            self._waiting_frames = self.av_eng.qimages_to_pixmaps(
                self._waiting_frames)
        log.info("Waiting message ready (%d frames).",
                 len(self._waiting_frames))

    def _on_greeting_ready(self):
        if self._greeting_frames:
            self._greeting_frames = self.av_eng.qimages_to_pixmaps(
                self._greeting_frames)
            self._start_playback(
                self._greeting_frames, self._greeting_fps,
                self._greeting_audio)
            self.replay_btn.setEnabled(True)

    def _on_waiting_video_done(self):
        """Waiting message finished — transition to response playback."""
        self._waiting_video_playing = False
        self._is_speaking = False
        self.voice.stop_playback()

        if self._pending_response_frames:
            self._play_response_video()
        else:
            self._start_thinking_animation()
            self._set_dot("● Generating …", "#e67e22")

    # ── Replay (greeting or last response)

    def _replay(self):
        if self._is_speaking or self._is_processing:
            return

        if self._last_response_frames and self._last_response_audio:
            self._start_playback(
                list(self._last_response_frames),
                self._last_response_fps,
                self._last_response_audio)
            return

        if self._greeting_frames and self._greeting_audio:
            self._start_playback(
                self._greeting_frames, self._greeting_fps,
                self._greeting_audio)

    # ── User input handling 

    def _on_send(self):
        text = self.text_input.text().strip()
        if not text or self._is_processing:
            return
        self.text_input.clear()
        self._add_chat_bubble(text, is_user=True)
        self._process_query(text)

    def _toggle_recording(self):
        if self._is_processing:
            return
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_mic_recording()

    def _start_mic_recording(self):
        self._is_recording = True
        self.mic_btn.setText("⏹")
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background: #c0392b; color: #fff;
                border: none; border-radius: 10px; font-size: 16px;
            }}
            QPushButton:hover {{ background: #e74c3c; }}
        """)
        self._set_dot("● Recording …", "#e74c3c")
        self.voice.start_recording()

    def _stop_recording(self):
        self._is_recording = False
        self.mic_btn.setText("🎤")
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_CARD}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; border-radius: 10px;
                font-size: 16px;
            }}
            QPushButton:hover {{ background: {COLOR_ACCENT}; color: #fff; }}
        """)
        self._set_dot("● Transcribing …", "#e67e22")

        audio_path = self.voice.stop_recording()
        if not audio_path:
            self._set_dot("● Idle", COLOR_SUBTEXT)
            return

        self._disable_input()

        def _transcribe():
            text = self.voice.transcribe(audio_path)
            if text:
                self.signals.transcription_done.emit(text)
            else:
                self.signals.system_message.emit(
                    "Could not transcribe audio — please try again.")
                self.signals.enable_input.emit()

        threading.Thread(target=_transcribe, daemon=True).start()

    def _on_transcription(self, text: str):
        self._add_chat_bubble(text, is_user=True)
        self._process_query(text)

    # ── Response pipeline (whole-response TTS + lip-sync) 

    def _stop_all_playback(self):
        """Stop any ongoing playback and reset state."""
        self.frame_timer.stop()
        self._audio_done_timer.stop()
        self.think_timer.stop()
        self.voice.stop_playback()
        self._is_speaking = False
        self._audio_only_playing = False
        self._waiting_video_playing = False

    def _process_query(self, question: str):
        self._stop_all_playback()

        self._is_processing = True
        self._query_id += 1
        qid = self._query_id

        self._disable_input()

        self._pending_response_frames = []
        self._pending_response_fps = 25.0
        self._pending_response_audio = ""
        self._pending_response_video = ""

        self._last_response_audio = ""
        self._last_response_frames = []
        self._last_response_fps = 25.0
        self._last_response_video = ""

        if (ENABLE_LIPSYNC and self.av_eng.wav2lip_available
                and self._waiting_frames and self._waiting_audio):
            self._waiting_video_playing = True
            self._start_playback(
                list(self._waiting_frames), self._waiting_fps,
                self._waiting_audio)
        else:
            self._waiting_video_playing = False
            self._start_thinking_animation()
            self._set_dot("● Thinking …", "#e67e22")
            self._update_status("Querying knowledge base …")

        self._current_bot_bubble = ChatBubble("", is_user=False)
        self.chat_layout.addWidget(self._current_bot_bubble)
        QTimer.singleShot(50, self._scroll_chat_to_bottom)

        def _producer():
            try:
                docs = self.kb.query(question, top_k=5)
                self.signals.status.emit("Generating response …")

                full_text = ""
                for token in self.llm.generate_response_stream(
                        question, docs):
                    if self._query_id != qid:
                        return
                    self.signals.bot_message_append.emit(token)
                    full_text += token

                if not full_text.strip():
                    self.signals.system_message.emit(
                        "No response generated.")
                    self.signals.status.emit("Ready")
                    self._is_processing = False
                    self.signals.enable_input.emit()
                    return

                if self._query_id != qid:
                    return

                tts_text = _clean_for_tts(full_text)
                self.signals.status.emit("Synthesizing speech …")
                audio_path = os.path.join(TEMP_DIR, "response.wav")
                audio_path = self.voice.synthesize(
                    tts_text, out_path=audio_path)

                if self._query_id != qid:
                    return

                if (ENABLE_LIPSYNC and self.av_eng.wav2lip_available
                        and audio_path):
                    self.signals.status.emit(
                        "Generating lip-sync video …")
                    video = \
                        self.av_eng.generate_talking_video_cached(
                            audio_path)
                    if video and self._query_id == qid:
                        video_audio = video.replace(
                            ".mp4", "_audio.wav")
                        if not os.path.isfile(video_audio):
                            self.av_eng.extract_audio_from_video(
                                video, video_audio)
                        qimages = \
                            self.av_eng.extract_frames_as_qimages(
                                video)
                        fps = self.av_eng.get_video_fps(video)
                        if qimages:
                            self._pending_response_frames = qimages
                            self._pending_response_fps = fps
                            self._pending_response_audio = (
                                video_audio
                                if os.path.isfile(video_audio)
                                else audio_path)
                            self._pending_response_video = video
                            self.signals.response_video_ready.emit()
                            return

                if audio_path and os.path.isfile(audio_path):
                    self._last_response_audio = audio_path
                    self.signals.speak_audio_only.emit(audio_path)

                self._is_processing = False
                self.signals.enable_input.emit()

            except Exception as exc:
                log.error("Response pipeline error: %s", exc)
                self.signals.system_message.emit(f"Error: {exc}")
                self.signals.status.emit("Ready")
                self._is_processing = False
                self.signals.enable_input.emit()

        threading.Thread(target=_producer, daemon=True).start()

    # ── Response video playback 

    def _on_response_video_ready(self):
        """Background thread finished generating the lip-sync video."""
        if self._pending_response_frames:
            self._pending_response_frames = \
                self.av_eng.qimages_to_pixmaps(
                    self._pending_response_frames)

        self._is_processing = False
        self.signals.enable_input.emit()

        if self._waiting_video_playing:
            return

        self._play_response_video()

    def _play_response_video(self):
        """Play the single whole-response lip-sync video."""
        if not self._pending_response_frames:
            self._update_status("Ready")
            return

        self.think_timer.stop()

        self._last_response_frames = list(
            self._pending_response_frames)
        self._last_response_fps = self._pending_response_fps
        self._last_response_audio = self._pending_response_audio
        self._last_response_video = self._pending_response_video

        self._pending_response_frames = []

        self._start_playback(
            list(self._last_response_frames),
            self._last_response_fps,
            self._last_response_audio)

    # ── Audio-only playback (greeting fallback) 

    def _play_audio_only(self, audio_path: str):
        self.think_timer.stop()
        self._set_speaking_style(True)
        self._set_dot("● Speaking", "#27ae60")
        self._update_status("Speaking …")

        self._audio_only_playing = True
        self._audio_start_time = time.time()

        self.voice.play_audio_nonblocking(audio_path)
        self._audio_done_timer.start(5000)

    def _on_audio_only_done(self):
        self._audio_only_playing = False
        if not self._is_speaking:
            self.avatar_label.setPixmap(self.av_eng.get_idle_pixmap())
            self._set_speaking_style(False)
            self._set_dot("● Idle", COLOR_SUBTEXT)
            self._update_status("Ready")
            self._update_replay_btn()

    # ── Streaming text to chat bubble 

    def _append_to_bot_bubble(self, token: str):
        if self._current_bot_bubble:
            self._current_bot_bubble.append_text(token)
            now = time.time()
            if now - getattr(self, '_last_scroll_t', 0) > 0.1:
                self._last_scroll_t = now
                QTimer.singleShot(30, self._scroll_chat_to_bottom)

    # ── Replay button 

    def _update_replay_btn(self):
        if self._last_response_frames and self._last_response_audio:
            self.replay_btn.setText("▶  Replay Last Response (lip-sync)")
            self.replay_btn.setEnabled(True)
        elif self._greeting_frames and self._greeting_audio:
            self.replay_btn.setText("▶  Replay Greeting")
            self.replay_btn.setEnabled(True)

    # ── Chat display 

    def _add_chat_bubble(self, text: str, is_user: bool = False):
        bubble = ChatBubble(text, is_user)
        self.chat_layout.addWidget(bubble)
        QTimer.singleShot(50, self._scroll_chat_to_bottom)

    def _add_system_message(self, text: str):
        msg = SystemMessage(text)
        self.chat_layout.addWidget(msg)
        QTimer.singleShot(50, self._scroll_chat_to_bottom)

    def _scroll_chat_to_bottom(self):
        vbar = self.chat_scroll.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    # ── Input enable / disable 

    def _disable_input(self):
        self.text_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        if self.voice.mic_available:
            self.mic_btn.setEnabled(False)

    def _enable_input(self):
        self.text_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        if self.voice.mic_available:
            self.mic_btn.setEnabled(True)
        self.text_input.setFocus()

    # ── Avatar frame playback 

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
            self.avatar_label.setStyleSheet(
                f"border: 3px solid #27ae60; border-radius: 14px; "
                f"background: {COLOR_CARD};")
            self._avatar_glow.setColor(QColor("#27ae60"))
            self._avatar_glow.setBlurRadius(50)
        else:
            self.avatar_label.setStyleSheet(
                f"border: 2px solid {COLOR_BORDER}; border-radius: 14px; "
                f"background: {COLOR_CARD};")
            self._avatar_glow.setColor(QColor(COLOR_ACCENT))
            self._avatar_glow.setBlurRadius(30)

    def _start_playback(self, frames: list, fps: float, audio_path: str):
        if not frames:
            return
        self._is_speaking = True
        self._audio_only_playing = False
        self._audio_done_timer.stop()
        self.think_timer.stop()
        self._frames = frames
        self._frame_idx = 0
        interval = max(1, int(1000 / fps))

        self._set_speaking_style(True)
        self._set_dot("● Speaking", "#27ae60")
        self._update_status("Speaking …")

        if audio_path and os.path.isfile(audio_path):
            self.voice.play_audio_nonblocking(audio_path)

        self.frame_timer.start(interval)

    def _next_frame(self):
        if self._frame_idx >= len(self._frames):
            self.frame_timer.stop()
            if self._waiting_video_playing:
                self._on_waiting_video_done()
                return
            self._is_speaking = False
            self.avatar_label.setPixmap(self.av_eng.get_idle_pixmap())
            self._set_speaking_style(False)
            self._set_dot("● Idle", COLOR_SUBTEXT)
            self._update_status("Ready")
            self._update_replay_btn()
            return

        frame = self._frames[self._frame_idx]
        if isinstance(frame, QImage):
            frame = self.av_eng._qimage_to_pixmap(frame)
            self._frames[self._frame_idx] = frame
        self.avatar_label.setPixmap(frame)
        self._frame_idx += 1

    # ── Status helpers 

    def _update_status(self, msg: str):
        self.status_bar.showMessage(msg)
        dot_map = {
            "speak":    ("● Speaking", "#27ae60"),
            "ready":    ("● Idle", COLOR_SUBTEXT),
            "generat":  ("● Generating …", "#e67e22"),
            "loading":  ("● Loading …", "#e67e22"),
            "initial":  ("● Initializing …", COLOR_ACCENT),
            "think":    ("● Thinking …", "#e67e22"),
            "query":    ("● Thinking …", "#e67e22"),
            "synthe":   ("● Generating …", "#e67e22"),
            "lip":      ("● Generating …", "#e67e22"),
            "transcri": ("● Transcribing …", "#e67e22"),
        }
        key = msg.lower()
        for k, (label, color) in dot_map.items():
            if k in key:
                self._set_dot(label, color)
                return

    def _set_dot(self, label: str, color: str):
        self.status_dot.setText(label)
        self.status_dot.setStyleSheet(
            f"color: {color}; font-size: 12px; background: transparent;")


# ── Entry point 

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
