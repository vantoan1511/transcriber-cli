import time

start = time.time()
import argparse
import os
import subprocess
import sys
import tempfile
import shutil

print("Started time:", time.time() - start)


def resource_path(relative_path):
    """Get absolute path to resource (for PyInstaller and dev)"""
    # If ffmpeg is in PATH, use it directly
    if relative_path.lower() == "ffmpeg.exe":
        ffmpeg_in_path = shutil.which("ffmpeg")
        if ffmpeg_in_path:
            return ffmpeg_in_path
        # Otherwise, look for ffmpeg.exe next to the executable
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, relative_path)


ffmpeg = resource_path("ffmpeg.exe")


def convert_audio_to_wav(input_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        try:
            subprocess.run([
                ffmpeg, "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", temp_wav_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("FFmpeg conversion failed")

        return temp_wav_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        sys.exit(1)


def format_srt(segments):
    def srt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, seg in enumerate(segments, 1):
        start, end = seg["timestamp"]
        text = seg["text"].strip()
        lines.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(lines)


def transcribe(audio_path, model_name, output_path, format):
    try:
        from transformers import pipeline
        start_time = time.time()
        print(f"Converting audio to 16kHz mono WAV...")
        wav_path = convert_audio_to_wav(audio_path)

        print(f"Loading model: {model_name} ...")
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            generate_kwargs={"task": "transcribe", "language": "vi"}
        )

        print(f"Transcribing...")
        result = asr(wav_path, return_timestamps=True)
        os.remove(wav_path)

        text = result.get("text", "").strip()
        segments = result.get("chunks", [])

        if format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif format == "srt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(format_srt(segments))

        print(f"Saved to: {output_path}")
        print(f"Duration: {time.time() - start_time}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using PhoWhisper")
    parser.add_argument("audio", help="Input audio file path")
    parser.add_argument("-m", "--model", default="vinai/PhoWhisper-tiny", help="PhoWhisper model to use")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-f", "--format", choices=["txt", "srt"], default="txt", help="Output format (txt or srt)")

    args = parser.parse_args()

    audio_path = args.audio
    output_path = args.output
    if not output_path:
        base = os.path.splitext(audio_path)[0]
        output_path = f"{base}.{args.format}"

    transcribe(audio_path, args.model, output_path, args.format)


if __name__ == "__main__":
    main()
