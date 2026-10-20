"""
Audio Transcription Pipeline.
Run: python transcribe.py
"""
import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_FILE_SIZE_MB = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file and print timestamped segments."
    )
    parser.add_argument(
        "audio_path", nargs="?",
        help="Path to audio file. If omitted, you will be prompted."
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)."
    )
    return parser.parse_args()


def validate_audio_path(raw_path):
    raw_path = raw_path.strip().strip('"').strip("'")
    path = Path(raw_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    if path.stat().st_size == 0:
        raise ValueError("The selected audio file is empty.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB; the prototype limit is {MAX_FILE_SIZE_MB} MB."
        )
    return path, size_mb


def transcribe_audio(audio_path, model_name):
    print(f"\nLoading model '{model_name}'...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    print("Transcribing audio. The first run may take longer while the model downloads.\n")
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=5,
    )

    print(
        f"Detected language: {info.language} "
        f"(probability: {info.language_probability:.2f})"
    )
    print("\n--- Timestamped transcript ---")

    transcript_parts = []
    segment_count = 0

    # Segments are yielded lazily, so they can be processed one at a time.
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        segment_count += 1
        transcript_parts.append(text)
        print(f"[{segment.start:8.2f}s - {segment.end:8.2f}s] {text}")

    if segment_count == 0:
        print("No speech segments were detected.")
        return

    print("\n--- Full transcript ---")
    print(" ".join(transcript_parts))
    print(f"\nCompleted: {segment_count} segments transcribed.")


def main():
    args = parse_args()
    raw_path = args.audio_path

    if not raw_path:
        raw_path = input("Enter the full path to your audio file: ")

    try:
        audio_path, size_mb = validate_audio_path(raw_path)
        print(f"Input file: {audio_path}")
        print(f"File size:  {size_mb:.2f} MB")
        transcribe_audio(audio_path, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nTranscription cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nTranscription failed ({type(exc).__name__}): {exc}\n"
            "Check the audio file and the model download. If it continues, "
            "share the complete error message.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
