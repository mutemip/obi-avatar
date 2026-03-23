# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Obi — Observability Assistant
Build on Windows with:  pyinstaller obi.spec
"""
import os
import sys

block_cipher = None
BASE = os.path.abspath(".")

a = Analysis(
    ["main.py"],
    pathex=[BASE],
    binaries=[],
    datas=[
        ("config.py",         "."),
        ("avatar_engine.py",  "."),
        ("voice_engine.py",   "."),
    ],
    hiddenimports=[
        "edge_tts",
        "edge_tts.communicate",
        "pydub",
        "cv2",
        "numpy",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "asyncio",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "pandas", "jupyter",
        "IPython", "notebook", "tkinter",
        "whisper", "sounddevice", "soundfile",
        "chromadb", "sentence_transformers",
        "ollama", "fitz",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Obi",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Obi",
)
