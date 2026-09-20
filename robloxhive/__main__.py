from __future__ import annotations

import argparse

from robloxhive import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="robloxhive")
    sub = parser.add_subparsers(dest="command")

    dashboard = sub.add_parser("dashboard", help="run the RobloxHive web dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.command == "dashboard":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit(
                "Dashboard dependencies are missing. Install with: pip install -e .[brain]"
            ) from exc
        uvicorn.run(
            "robloxhive.interface.dashboard:create_app",
            host=args.host,
            port=args.port,
            factory=True,
        )
        return

    print(f"RobloxHive {__version__}")
    print("Run 'python -m robloxhive dashboard' to start the dashboard.")


if __name__ == "__main__":
    main()
