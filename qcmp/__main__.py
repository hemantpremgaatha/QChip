import argparse
import asyncio
import logging

from .processor import QuantumMusicProcessor


def main():
    ap = argparse.ArgumentParser(description="Quantum Computing Music Processor")
    ap.add_argument("--source", choices=["synthetic", "mic", "termux"], default="synthetic")
    ap.add_argument("--chunks", type=int, default=100, help="chunks to process (~23 ms each)")
    ap.add_argument("--fast", action="store_true", help="do not pace to real time")
    ap.add_argument("--config")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    proc =QuantumMusicProcessor(a.config, a.source)
    asyncio.run(proc.process_audio(a.chunks, realtime=not a.fast))


if __name__ == "__main__":
    main()
