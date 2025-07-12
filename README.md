# Transcriber CLI - Transcription Tool v1.0

A command-line tool for transcribing Vietnamese audio using PhoWhisper models via Hugging Face Transformers.

## Features

- Converts audio to 16kHz mono WAV using FFmpeg
- Transcribes audio to text or SRT subtitle format (currently not working)
- Supports different PhoWhisper models

## Requirements

- Python 3.8+
- [transformers](https://pypi.org/project/transformers/)
- FFmpeg (must be in your PATH or in the project directory)
