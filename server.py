"""
Whisper speed/accuracy test bench.

Backend: whisper.cpp (C++ inference on the pretrained OpenAI Whisper weights,
converted to ggml) via the `pywhispercpp` Python bindings.

Run:
    python server.py            then open http://localhost:8000
"""

import csv
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from pywhispercpp.model import Model

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"          # ggml weights are downloaded here on first use
RESULTS_CSV = ROOT / "results.csv"    # one row per Yes/No answer
HINTS_FILE = ROOT / "hints.txt"       # passed to whisper as its initial prompt
SAMPLE_RATE = 16000                   # whisper expects 16 kHz mono float32

# Smallest -> largest. ".en" models are English-only and are usually a bit more
# accurate than the multilingual model of the same size on English speech.
# "-q5_0"/"-q8_0" are quantized versions: smaller/faster, small accuracy cost.
MODEL_CHOICES = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large-v3-turbo-q5_0",
    "large-v3-turbo",
    "large-v3",
]
DEFAULT_MODEL = "base.en"

# whisper.cpp scales with physical cores; more threads than that tends to hurt.
N_THREADS = min(8, os.cpu_count() or 4)

app = FastAPI()

# whisper.cpp contexts are not thread-safe, and we only keep one model in
# memory at a time (large-v3 is ~3 GB), so all model access goes through this lock.
_lock = threading.Lock()
_model: Model | None = None
_model_key: tuple[str, bool] | None = None   # (name, beam_search)
_load_seconds = 0.0


def load_model(name: str, beam_search: bool) -> float:
    """Load (or reuse) a model. Returns load time in seconds."""
    global _model, _model_key, _load_seconds
    if (name, beam_search) == _model_key:
        return 0.0

    _model = None  # free the old one before allocating the new one
    t0 = time.perf_counter()

    # ---------------------------------------------------------------------
    # MODEL INITIALIZATION (Windows / Linux, CPU)
    #
    # The PyPI wheel of pywhispercpp is a CPU-only build. For an NVIDIA GPU on
    # Windows/Linux you would rebuild it with:
    #     GGML_CUDA=1 pip install git+https://github.com/absadiki/pywhispercpp
    # and nothing below would need to change (use_gpu defaults to True).
    # ---------------------------------------------------------------------
    _model = Model(
        name,
        models_dir=str(MODELS_DIR),
        # 0 = greedy decoding (fastest). 1 = beam search (slower, usually a
        # little more accurate -- worth testing for the long-answer section).
        params_sampling_strategy=1 if beam_search else 0,
        n_threads=N_THREADS,
        language="en",
        print_progress=False,
        print_realtime=False,
        redirect_whispercpp_logs_to=None,   # silence whisper.cpp's stderr spam
    )

    # ---------------------------------------------------------------------
    # MODEL INITIALIZATION ON A MACBOOK (Apple Silicon, M1 or newer)
    #
    # 1) Install pywhispercpp from source so whisper.cpp is compiled on the
    #    Mac with Metal (GPU) enabled -- Metal is on by default for macOS builds:
    #        pip install git+https://github.com/absadiki/pywhispercpp
    #
    #    Optional, recommended for speed: also enable Core ML, which runs the
    #    encoder (the expensive half of Whisper) on the Apple Neural Engine:
    #        WHISPER_COREML=1 pip install git+https://github.com/absadiki/pywhispercpp
    #
    # 2) If you enabled Core ML, generate the Core ML encoder once per model
    #    with the script in the whisper.cpp repo (needs `pip install ane_transformers
    #    openai-whisper coremltools`):
    #        ./models/generate-coreml-model.sh base.en
    #    and copy the resulting `ggml-base.en-encoder.mlmodelc` folder into
    #    MODELS_DIR next to `ggml-base.en.bin`. whisper.cpp picks it up
    #    automatically; the first run is slow while macOS compiles it for the ANE.
    #
    # 3) The init call then looks like this:
    #
    #    _model = Model(
    #        name,
    #        models_dir=str(MODELS_DIR),
    #        params_sampling_strategy=1 if beam_search else 0,
    #        # Use only the performance cores (e.g. 4 on M1/M2, 6+ on Pro/Max).
    #        # Efficiency cores slow whisper.cpp down. Check with:
    #        #     sysctl hw.perflevel0.physicalcpu
    #        n_threads=4,
    #        language="en",
    #        print_progress=False,
    #        print_realtime=False,
    #        redirect_whispercpp_logs_to=None,
    #        context_params={
    #            "use_gpu": True,      # Metal (default True, explicit for clarity)
    #            "flash_attn": True,   # faster attention on Metal
    #        },
    #    )
    #
    # 4) Confirm the backends are active by checking the output of
    #    `Model.system_info()` for "METAL = 1" (and "COREML = 1" if enabled).
    # ---------------------------------------------------------------------

    _model_key = (name, beam_search)
    _load_seconds = time.perf_counter() - t0
    return _load_seconds


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/models")
def models():
    return {
        "choices": MODEL_CHOICES,
        "current": _model_key[0] if _model_key else None,
        "beam_search": _model_key[1] if _model_key else False,
        "load_seconds": _load_seconds,
        "n_threads": N_THREADS,
    }


class LoadRequest(BaseModel):
    name: str
    beam_search: bool = False


@app.post("/api/model")
def set_model(req: LoadRequest):
    if req.name not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model {req.name}")
    # First use of a model also downloads it (75 MB for tiny up to ~3 GB for
    # large-v3), so the very first load time includes the download.
    with _lock:
        seconds = load_model(req.name, req.beam_search)
    return {"name": req.name, "beam_search": req.beam_search, "load_seconds": seconds}


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Body: raw little-endian float32 PCM, 16 kHz mono (resampled in the browser)."""
    audio = np.frombuffer(await request.body(), dtype=np.float32)
    if audio.size < SAMPLE_RATE // 4:
        raise HTTPException(400, "Recording too short")
    return await run_in_threadpool(_transcribe_sync, audio)


def read_hints() -> str:
    """Non-comment lines of hints.txt joined into one prompt. Read on every call so edits apply live."""
    if not HINTS_FILE.exists():
        return ""
    lines = HINTS_FILE.read_text(encoding="utf-8").splitlines()
    return " ".join(l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#"))


DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalize_digits(text: str) -> str:
    """Spoken digit sequence -> plain digit string: "Four, one. 3-5" -> "4135".

    Number words are mapped to digits, then everything that isn't a digit
    (punctuation, spaces, filler words) is dropped.
    """
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return "".join(DIGIT_WORDS.get(w, w if w.isdigit() else "") for w in words)


def _transcribe_sync(audio: np.ndarray):
    hints = read_hints()
    with _lock:
        if _model is None:
            load_model(DEFAULT_MODEL, False)
        t0 = time.perf_counter()
        # Always pass initial_prompt: pywhispercpp keeps params between calls,
        # so an empty string is what clears hints after the file is emptied.
        segments = _model.transcribe(audio, initial_prompt=hints)
        elapsed = time.perf_counter() - t0
        name, beam = _model_key
    audio_seconds = audio.size / SAMPLE_RATE
    raw = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text": normalize_digits(raw),
        "raw_text": raw,   # Whisper's output before normalize_digits

        "model": name,
        "beam_search": beam,
        "audio_seconds": round(audio_seconds, 2),
        "inference_seconds": round(elapsed, 3),
        # real-time factor: <1 means faster than real time
        "rtf": round(elapsed / audio_seconds, 3),
    }


class Feedback(BaseModel):
    text: str
    correct: bool
    model: str
    beam_search: bool
    audio_seconds: float
    inference_seconds: float
    roundtrip_seconds: float   # stop-recording -> transcript shown, measured in browser


FIELDS = list(Feedback.model_fields) + ["timestamp"]


@app.post("/api/feedback")
def feedback(fb: Feedback):
    new_file = not RESULTS_CSV.exists()
    fields = FIELDS
    if not new_file:
        # Keep appending in the existing file's column order (older files
        # also have a "prompt" column, which is left blank).
        with RESULTS_CSV.open(encoding="utf-8") as f:
            fields = next(csv.reader(f), FIELDS)
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({**fb.model_dump(), "timestamp": datetime.now().isoformat(timespec="seconds")})
    return {"ok": True}


@app.get("/api/stats")
def stats():
    """Per model/decoding: number of trials, % correct, mean inference time and RTF."""
    if not RESULTS_CSV.exists():
        return []
    groups: dict[str, list[dict]] = {}
    with RESULTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["model"] + (" (beam)" if row["beam_search"] == "True" else "")
            groups.setdefault(key, []).append(row)
    out = []
    for key, rows in groups.items():
        inf = [float(r["inference_seconds"]) for r in rows]
        aud = [float(r["audio_seconds"]) for r in rows]
        out.append({
            "model": key,
            "trials": len(rows),
            "accuracy": sum(r["correct"] == "True" for r in rows) / len(rows),
            "mean_inference_seconds": sum(inf) / len(inf),
            "mean_rtf": sum(i / a for i, a in zip(inf, aud)) / len(rows),
        })
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
