# Audio Transcription Pipeline

A small Python pipeline that accepts an audio file, transcribes speech with `faster-whisper`, and prints timestamped segments and the full transcript in the terminal. It does not need a web server or user interface.

## Requirements
- Python 3.13 recommended
- Internet connection on the first run to download the speech model
- CPU supported; speed depends on the computer and model size

## Setup in VS Code on Windows

Activate the project's existing environment:

```cmd
.venv\Scripts\activate
```

If you have not created it yet:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

## Run the program

Option A — be prompted for the audio path:

```cmd
python transcribe.py
```

Paste the full path when asked, for example:

```text
C:\Users\YourName\Downloads\english_practice.mp3
```

Replace the example with the actual path on your computer.

Option B — pass the path directly:

```cmd
python transcribe.py "C:\Users\YourName\Downloads\english_practice.mp3"
```

To use a smaller, faster model:

```cmd
python transcribe.py "C:\path\to\audio.mp3" --model tiny
```

Available models: `tiny`, `base` (default), `small`, `medium`, `large-v3`. Larger models usually require more memory and take longer on CPU.

## Design decisions
- Validates that the path exists, is a file, has a supported extension, is non-empty, and is no larger than 100 MB.
- Uses `faster-whisper` to decode common audio formats and transcribe speech.
- Prints start/end timestamps for every returned speech segment.
- Processes segments as an iterator instead of first storing all segment objects in memory. The complete text is also collected to print a final transcript.
- Loads the model once per program run.
- Reports invalid input and transcription errors in the terminal.
- Supports local single-user execution without a server, endpoint, or UI.

## Limitations and future improvements
This is a prototype, not a multi-user production service. The 100 MB limit is a simple guard. The script prints results but does not save them to a file. A production system could add automated tests, structured logs, output persistence, bounded retries for transient errors, progress tracking, and a background queue for concurrent jobs.
