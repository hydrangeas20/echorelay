"""``python -m echorelay`` — run the server with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoRelay streaming STT server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":
    main()
