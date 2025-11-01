import os
import sys
import json
import logging
import shutil
import subprocess
import tracemalloc
import gc
import psutil
import torch
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm


# Global Configuration

LANGUAGE_CODE = "hi"       # Default language code
AUDIO_PATH = "input.wav"   # Default audio file path
MODEL_SIZE = "large-v3"
USE_GPU = True


# Logging Configuration

logging.basicConfig(
    filename="debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started")


# Cache and Dependencies

cacheDir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cacheDir, exist_ok=True)

ffmpegPath = "ffmpeg.exe"
try:
    subprocess.run([ffmpegPath, "-version"], capture_output=True, check=True)
    AudioSegment.ffmpeg = ffmpegPath
    print(f"ffmpeg found at {ffmpegPath}")
except Exception:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpegPath = "ffmpeg"
        AudioSegment.ffmpeg = ffmpegPath
        print("ffmpeg found in system PATH")
    except Exception as e:
        logging.error(f"ffmpeg not found: {str(e)}")
        print("ffmpeg not found. Please install and add to PATH.")
        sys.exit(1)


# Cache Setup

try:
    if os.path.exists(cacheDir):
        shutil.rmtree(cacheDir)
        print(f"Cleared existing cache directory {cacheDir}")
    os.makedirs(cacheDir, exist_ok=True)
except Exception as e:
    logging.error(f"Failed to clear cache directory {cacheDir}: {str(e)}")
    sys.exit(1)

# Verify write permissions and disk space
try:
    testFile = os.path.join(cacheDir, "test.txt")
    with open(testFile, "w") as f:
        f.write("test")
    os.remove(testFile)

    diskUsage = shutil.disk_usage(os.getcwd())
    if diskUsage.free < 5e9:  # 5 GB threshold
        print(f"Insufficient disk space: {diskUsage.free / 1e9:.2f} GB available")
        sys.exit(1)
except Exception as e:
    logging.error(f"Cache directory issue: {str(e)}")
    sys.exit(1)


# GPU / CPU Setup

device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
computeType = "float32" if device == "cuda" else "int8"

if device == "cuda":
    try:
        torch.cuda.init()
        gpuInfo = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpuInfo.name}, VRAM: {gpuInfo.total_memory / 1e9:.2f} GB")
    except Exception as e:
        logging.error(f"CUDA initialization failed: {str(e)}")
        print("Falling back to CPU.")
        device = "cpu"
        computeType = "int8"


# Load Model

gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

model = None
try:
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=computeType)
    print(f"Model {MODEL_SIZE} loaded on {device}")
except Exception as e:
    logging.error(f"Model load failed on {device}: {str(e)}")
    print(f"Model load failed on {device}: {str(e)}")
    if device == "cuda":
        device = "cpu"
        computeType = "int8"
        try:
            model = WhisperModel(MODEL_SIZE, device=device, compute_type=computeType)
            print(f"Model {MODEL_SIZE} loaded on CPU with {computeType}")
        except Exception as e:
            logging.error(f"Model load failed on CPU: {str(e)}")
            sys.exit(1)
    else:
        sys.exit(1)

if model is None or not hasattr(model, "transcribe"):
    logging.error("Model is not ready for transcription.")
    print("Model is not ready for transcription.")
    sys.exit(1)


# Functions

def splitAudio(audioPath=AUDIO_PATH, chunkLengthMs=20000):
    logging.info(f"Splitting {audioPath} into chunks of {chunkLengthMs} ms")
    chunks = []
    try:
        tracemalloc.start()
        tempAudioPath = os.path.join(os.getcwd(), "temp_input.wav")
        ffmpegCmd = [
            ffmpegPath, "-i", audioPath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-y", tempAudioPath
        ]
        result = subprocess.run(ffmpegCmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"ffmpeg failed for {audioPath}: {result.stderr}")
            return chunks
        
        audio = AudioSegment.from_file(tempAudioPath)
        durationMs = len(audio)
        if durationMs % chunkLengthMs != 0:
            paddingMs = chunkLengthMs - (durationMs % chunkLengthMs)
            audio += AudioSegment.silent(duration=paddingMs)
        
        tempDir = Path(os.path.join(os.getcwd(), "temp_chunks"))
        tempDir.mkdir(exist_ok=True)

        for i in range(0, len(audio), chunkLengthMs):
            chunk = audio[i:i + chunkLengthMs]
            chunkPath = tempDir / f"{Path(audioPath).stem}_chunk{i//1000}.wav"
            try:
                chunk.export(chunkPath, format="wav")
                chunks.append((str(chunkPath), i / 1000.0))
            except Exception as e:
                logging.error(f"Failed to export chunk {chunkPath}: {str(e)}")

        if Path(tempAudioPath).exists():
            os.remove(tempAudioPath)
        
        tracemalloc.stop()
        return chunks
    except Exception as e:
        logging.error(f"Error splitting audio {audioPath}: {str(e)}")
        return chunks


def transcribeAudio(chunkPath, model, device, language=LANGUAGE_CODE):
    logging.info(f"Transcribing {chunkPath} in language {language}")
    if not Path(chunkPath).is_file():
        logging.error(f"Chunk file not found: {chunkPath}")
        return {"error": f"Chunk file not found: {chunkPath}"}
    try:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        segments, info = model.transcribe(
            chunkPath,
            language=language,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            word_timestamps=True
        )

        transcriptSegments = []
        fullText = ""
        wordTimestamps = []

        for segment in segments:
            transcriptSegments.append({
                "start": int(segment.start * 1000),
                "end": int(segment.end * 1000),
                "text": segment.text.strip()
            })
            fullText += segment.text.strip() + " "
            for word in segment.words:
                wordTimestamps.append({
                    "word": word.word.strip(),
                    "start": int(word.start * 1000),
                    "end": int(word.end * 1000)
                })

        return {
            "segments": transcriptSegments,
            "full_text": fullText.strip(),
            "word_timestamps": wordTimestamps,
            "language": info.language,
            "confidence": info.language_probability
        }
    except Exception as e:
        logging.error(f"Transcription failed for {chunkPath}: {str(e)}")
        return {"error": str(e)}


# Main Execution

if __name__ == "__main__":
    if not os.path.isfile(AUDIO_PATH):
        logging.error(f"Audio file not found: {AUDIO_PATH}")
        print(f"Audio file not found: {AUDIO_PATH}")
        sys.exit(1)

    chunks = splitAudio(AUDIO_PATH)
    if not chunks:
        logging.error("No chunks created. Exiting.")
        print("No chunks created. Exiting.")
        sys.exit(1)

    results = []
    for chunkPath, offset in tqdm(chunks, desc="Transcribing chunks"):
        result = transcribeAudio(chunkPath, model, device, LANGUAGE_CODE)
        if "error" not in result:
            result["offset"] = offset
            results.append(result)
        else:
            logging.error(f"Failed to transcribe chunk {chunkPath}: {result['error']}")

    outputFile = "transcription_results.json"
    try:
        with open(outputFile, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Transcription results saved to {outputFile}")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        print(f"Failed to save results: {str(e)}")
        sys.exit(1)

    tempDir = Path(os.path.join(os.getcwd(), "temp_chunks"))
    if tempDir.exists():
        shutil.rmtree(tempDir)
        print(f"Removed temporary directory: {tempDir}")
