"""Bluetooth audio recorder with Deepgram transcription.

Records incoming A2DP audio from the Windows target machine
(paired as "Office Speaker") and optionally streams to Deepgram
for real-time transcription.

Audio files saved to: data/audio/
Transcripts saved to: data/audio/transcripts/

Usage:
    # Record only (no transcription)
    uv run python scripts/bt_audio_recorder.py

    # Record + live Deepgram transcription
    uv run python scripts/bt_audio_recorder.py --transcribe

    # List available Bluetooth audio sources
    uv run python scripts/bt_audio_recorder.py --list-sources
"""

import argparse
import json
import subprocess
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
TRANSCRIPT_DIR = AUDIO_DIR / "transcripts"


def get_deepgram_key() -> str:
    """Retrieve Deepgram API key from pass store."""
    result = subprocess.run(
        ["pass", "show", "api/deepgram"],
        capture_output=True, text=True,
    )
    key = result.stdout.strip().split("\n")[0]
    if not key:
        print("ERROR: Deepgram API key not found in pass store (api/deepgram)")
        sys.exit(1)
    return key


def list_bt_sources():
    """List available Bluetooth audio sources (PipeWire)."""
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True,
    )
    bt_sources = []
    for line in result.stdout.strip().split("\n"):
        if "bluez" in line.lower():
            bt_sources.append(line)

    if bt_sources:
        print("Bluetooth audio sources:")
        for s in bt_sources:
            print(f"  {s}")
    else:
        print("No Bluetooth audio sources found.")
        print("Make sure Windows is connected and streaming audio.")
        print("Run: pactl list sources short | grep -i blue")
    return bt_sources


def find_bt_source() -> str | None:
    """Find the first Bluetooth audio source."""
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and "bluez" in parts[1].lower() and "input" in parts[1].lower():
            return parts[1]  # source name
    return None


def record_audio(source: str, output_path: Path, duration: int = 0):
    """Record audio from a PipeWire/PulseAudio source.

    Args:
        source: PipeWire source name (e.g., bluez_input.XX_XX_XX_XX)
        output_path: Path to write WAV file
        duration: Max seconds (0 = until interrupted)
    """
    cmd = [
        "pw-record",
        f"--target={source}",
        "--format=s16",
        "--rate=16000",
        "--channels=1",
        str(output_path),
    ]
    if duration > 0:
        cmd.insert(1, f"--timeout={duration}")

    print(f"Recording from {source} → {output_path}")
    print("Press Ctrl+C to stop recording")

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    return output_path


def transcribe_file(audio_path: Path, api_key: str) -> dict:
    """Send audio file to Deepgram for transcription."""
    import urllib.request

    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&diarize=true"

    audio_data = audio_path.read_bytes()
    req = urllib.request.Request(
        url,
        data=audio_data,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
        method="POST",
    )

    print(f"Transcribing {audio_path.name} ({len(audio_data)/1024:.1f}KB)...")
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    return result


def main():
    parser = argparse.ArgumentParser(description="Bluetooth audio recorder + Deepgram")
    parser.add_argument("--transcribe", action="store_true",
                        help="Transcribe audio with Deepgram after recording")
    parser.add_argument("--list-sources", action="store_true",
                        help="List available Bluetooth audio sources")
    parser.add_argument("--duration", type=int, default=0,
                        help="Max recording duration in seconds (0=manual stop)")
    parser.add_argument("--source", type=str, default="",
                        help="Specific PipeWire source name (auto-detect if empty)")
    args = parser.parse_args()

    if args.list_sources:
        list_bt_sources()
        return

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Find Bluetooth source
    source = args.source or find_bt_source()
    if not source:
        print("No Bluetooth audio source found!")
        print("Steps to fix:")
        print("  1. Run: python scripts/bt_speaker_agent.py")
        print("  2. On Windows: Settings > Bluetooth > Add device > 'WF-1000XM5'")
        print("  3. Set 'WF-1000XM5' as Windows audio output")
        print("  4. Run this script again")
        sys.exit(1)

    # Record
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audio_path = AUDIO_DIR / f"bt_{timestamp}.wav"

    record_audio(source, audio_path, duration=args.duration)
    print(f"\nRecorded: {audio_path} ({audio_path.stat().st_size / 1024:.1f}KB)")

    # Transcribe if requested
    if args.transcribe:
        api_key = get_deepgram_key()
        result = transcribe_file(audio_path, api_key)

        # Extract transcript text
        transcript = ""
        alternatives = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])
        )
        if alternatives:
            transcript = alternatives[0].get("transcript", "")

        # Save transcript
        transcript_path = TRANSCRIPT_DIR / f"bt_{timestamp}.json"
        with open(transcript_path, "w") as f:
            json.dump({
                "audio_file": audio_path.name,
                "timestamp": timestamp,
                "transcript": transcript,
                "deepgram_result": result,
            }, f, indent=2)

        print(f"Transcript: {transcript_path}")
        if transcript:
            print(f"\n--- Transcript ---")
            print(transcript[:500])
            if len(transcript) > 500:
                print(f"... ({len(transcript)} chars total)")
        else:
            print("(no speech detected)")


if __name__ == "__main__":
    main()
