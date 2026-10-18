from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def server_state(port):
    """Distinguish our existing server from an unrelated port occupant."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
        return "available"
    except OSError:
        try:
            # Loopback health checks must not go through a system HTTP proxy.
            with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
                health = json.loads(response.read(4096))
            if health.get("name") == "RealityForge" and health.get("status") == "ok":
                return "running"
        except (OSError, URLError, ValueError, AttributeError):
            pass
        return "occupied"


def main():
    parser = argparse.ArgumentParser(description="RealityForge: bounding box -> 3D model -> schematic -> Minecraft Java world")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Launch the local browser interface")
    serve.add_argument("--port", type=int, default=8000)
    build = commands.add_parser("generate", help="Generate from a bounding box or offline demo")
    build.add_argument("--bbox", help="west,south,east,north (use --bbox=-79.4,43.65,-79.397,43.652 for negatives)")
    build.add_argument("--demo", action="store_true")
    build.add_argument("--terrain", action="store_true")
    build.add_argument("--scale", type=int, choices=[1, 2, 4, 8], default=1, help="Metres per block (use 8 for a 5 km overview)")
    build.add_argument("--name", default="RealityForge")
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("Port must be between 1 and 65535.")
        url = f"http://127.0.0.1:{args.port}/"
        state = server_state(args.port)
        if state == "running":
            print(f"RealityForge is already running. Open {url}", flush=True)
            print("No second server was started. You can close this window.", flush=True)
            return
        if state == "occupied":
            parser.exit(2, f"Port {args.port} is occupied by another app or an unresponsive server.\n"
                        "Try: npm run dev -- --port 8001\n")
        print(f"Starting RealityForge at {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
        import uvicorn
        uvicorn.run("forge.app:app", host="127.0.0.1", port=args.port)
        return
    from .geography import DEMO_CENTER, centered_bbox, validate_bbox
    from .pipeline import generate
    if not args.bbox and not args.demo:
        parser.error("Provide --bbox or --demo.")
    try:
        bbox = validate_bbox([float(v) for v in args.bbox.split(",")] if args.bbox else centered_bbox(*DEMO_CENTER))
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new folder.")
    result = generate({"name": args.name, "bbox": bbox, "source": "demo" if args.demo else "live",
                       "meters_per_block": args.scale, "terrain": args.terrain, "photos": []},
                      args.output, Path(__file__).resolve().parents[1] / "data" / "cache",
                      lambda percent, stage, message: print(f"[{percent:3}%] {message}", flush=True))
    print(json.dumps(result["report"]["statistics"], indent=2))
    print(f"Ready: {args.output.resolve() / 'RealityForge-world.zip'}")


if __name__ == "__main__":
    main()
