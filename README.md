# audio-transcriber-python

# 🎧 High-Accuracy Audio Transcription (Faster Whisper + FFmpeg)

A professional, GPU-optimized Python script for **speech-to-text transcription** using the **Faster-Whisper** model.  
It automatically splits, converts, and transcribes long audio files with accuracy and reliability.

---

## 🚀 Features

✅ Uses **Faster-Whisper** for ultra-fast transcription  
✅ Automatically converts audio using **FFmpeg**  
✅ Splits long audio files into smaller chunks for efficiency  
✅ Supports **GPU (CUDA)** and **CPU** fallback  
✅ Saves output to `transcription_results.json`  
✅ Logs every step for debugging and monitoring  

---

## 🧰 Requirements

### 1. Install Required Python Packages
Install all dependencies via pip:

```bash
pip install faster-whisper==1.0.3 librosa==0.10.2.post1 psutil==6.0.0 tqdm==4.66.5 numpy==1.26.4 tokenizers==0.19.1 torch
