"""CLI entry point for the Quantum Computing Music Processor (QCMP)."""
from __future__ import annotations

import argparse
import asyncio
import logging

from qcmp.music_processor import QuantumMusicProcessor, SyntheticSource, WavFileSource


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum Computing Music Processor")
    parser.add_argument("--config", default=None, help="Path to config.ini")
    parser.add_argument("--source", choices=["mic", "file", "synthetic"], default="mic")
    parser.add_argument("--wav", help="WAV file path when --source file")
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="Stop after N audio chunks (useful for testing)"
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    args = build_arg_parser().parse_args()

    audio_source = None
    if args.source == "file":
        if not args.wav:
            raise SystemExit("--wav is required when --source file")
        audio_source = WavFileSource(args.wav)
    elif args.source == "synthetic":
        audio_source = SyntheticSource(rate=44100)

    processor = QuantumMusicProcessor(config_path=args.config, audio_source=audio_source)
    asyncio.run(processor.process_audio(max_chunks=args.max_chunks))


if __name__ == "__main__":
    main()
