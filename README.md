# Whisper test bench

whisper.cpp (via `pywhispercpp`) + a small browser UI that loops: record → "Is this what you said?" → Yes/No.
Every answer is appended to `results.csv`. Model weights are downloaded to `models/` on first use.

## Windows laptop (runs in WSL)

Smart App Control is on, and it blocks the unsigned whisper.cpp binary that pywhispercpp ships for Windows, so the server runs in WSL. The browser still runs on Windows.

### Start

1. Open PowerShell or the VS Code terminal and run:

   ```powershell
   wsl -d Ubuntu --cd "/mnt/c/Users/dvck/XR Lab/Whispr-Test" -- ~/whispr-venv/bin/python server.py
   ```

2. Wait for `Uvicorn running on http://127.0.0.1:8000`.
3. Open **http://localhost:8000** in Chrome or Edge.

Leave that terminal open, because closing it stops the server.

### Stop

Press **Ctrl+C** in the terminal where the server is running.

If you can't find that terminal, run this from any terminal:

```powershell
wsl -d Ubuntu -- pkill -f whispr-venv/bin/python
```

### Restart

Stop, then start again. When you need to:

| You changed | What to do |
|---|---|
| `hints.txt` | Nothing. It's re-read on every recording. |
| `index.html` | Refresh the browser. |
| `server.py` | Restart the server. |

### Troubleshooting

- **`address already in use` on start**: a server is already running. Use it, or stop it with the `pkill` command above and start again.
- **Page won't load**: check that the server terminal shows `Uvicorn running` and hasn't exited with an error.

### One-time setup (already done on this laptop)

Run inside WSL (`wsl -d Ubuntu`):

```bash
cd "/mnt/c/Users/dvck/XR Lab/Whispr-Test"
python3 -m venv --without-pip ~/whispr-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ~/whispr-venv/bin/python
~/whispr-venv/bin/pip install -r requirements.txt
```

## Mac

### One-time setup

```bash
cd path/to/Whispr-Test
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn numpy
pip install git+https://github.com/absadiki/pywhispercpp   # builds with Metal
```

See the comments in `load_model()` in `server.py` for the Metal and Core ML setup.

### Start

```bash
cd path/to/Whispr-Test
source venv/bin/activate
python server.py
```

Then open **http://localhost:8000**.

### Stop

Press **Ctrl+C** in that terminal. If you can't find it, run `pkill -f "python server.py"`.

## Using it

- Pick a model, tick **beam search** if you want it, then click **Load model**. The load time is shown, and the first load of a model includes its download.
- Press **Space** to start and stop recording. Press **Y** or **N** to answer.
- `hints.txt` steers Whisper toward a format (currently digit-only numbers). The big number on the confirm screen is the cleaned-up result, and the small line under it is Whisper's original output.
- **Inference** is time spent inside whisper.cpp. **Round trip** runs from when you stop recording to when the transcript appears. **RTF** is inference time ÷ audio length, so lower is faster.
- Timings here are only relative. CPU timings on WSL won't match a MacBook with Metal or Core ML.
