from __future__ import annotations

import argparse
import os

from robloxhive import __version__


def _run_dashboard(args: argparse.Namespace) -> None:
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


def _run_body(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise SystemExit("The Roblox Body Node is Windows-only.")

    from robloxhive.body.bridge import BodyBridge
    from robloxhive.body.discovery import discover_roblox_windows
    from robloxhive.body.factory import create_generic_skill_executor

    windows = discover_roblox_windows()

    if args.list_windows:
        if not windows:
            print("No Roblox windows detected.")
            return
        for item in windows:
            print(f"PID={item.pid} HWND={item.hwnd} TITLE={item.title!r}")
        return

    if args.pid is not None:
        matches = [item for item in windows if item.pid == args.pid]
        if len(matches) != 1:
            raise SystemExit(f"Could not find exactly one Roblox window for PID {args.pid}.")
        instance = matches[0]
    else:
        if len(windows) != 1:
            raise SystemExit(
                "More than one (or zero) Roblox window detected. "
                "Run with --list-windows, then pass the bot PID explicitly with --pid."
            )
        instance = windows[0]

    def build_executor(game_id: int):
        return create_generic_skill_executor(
            hwnd=instance.hwnd,
            game_id=game_id,
            template_root=args.template_root,
            model_path=args.model,
            labels_path=args.labels,
            enable_ocr=not args.no_ocr,
            input_mode=args.input_mode,
        )

    executor = build_executor(args.game_id)
    bridge = BodyBridge(
        brain_url=args.brain_url,
        executor=executor,
        agent_id=args.agent_id,
        metadata={
            "pid": instance.pid,
            "hwnd": instance.hwnd,
            "title": instance.title,
            "game_id": args.game_id,
            "input_backend": args.input_mode,
        },
        executor_factory=build_executor,
    )

    print(f"RobloxHive Body {__version__}")
    print(f"Agent   : {args.agent_id}")
    print(f"PID/HWND: {instance.pid}/{instance.hwnd}")
    print(f"Game ID : {args.game_id} (0 = pilih dari dashboard)")
    print(f"Input   : {args.input_mode}")
    print(f"Brain   : {args.brain_url}")
    print(f"Skills  : {', '.join(executor.available())}")
    print(f"Perception: {executor.describe()['metadata'].get('perception', {})}")
    bridge.run_forever()


def _run_mm2_train(args: argparse.Namespace) -> None:
    import json
    from robloxhive.games.murder_mystery_2.training import train_mm2_detector

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    result = train_mm2_detector(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        output_dir=args.output_dir,
        model_output_dir=args.model_output_dir,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="robloxhive")
    sub = parser.add_subparsers(dest="command")

    dashboard = sub.add_parser("dashboard", help="run the RobloxHive web dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    body = sub.add_parser("body", help="run the Windows Roblox Body Node")
    body.add_argument("--brain-url", default="http://127.0.0.1:8765")
    body.add_argument("--agent-id", default="agent-01")
    body.add_argument("--game-id", type=int, default=0)
    body.add_argument("--pid", type=int)
    body.add_argument(
        "--input-mode",
        choices=["auto", "foreground", "message"],
        default="auto",
        help="auto/foreground uses guarded SendInput; message is background experimental",
    )
    body.add_argument("--template-root", default="data/templates")
    body.add_argument("--model", help="optional YOLOv8-style ONNX detector path")
    body.add_argument("--labels", help="optional labels.txt or labels.json path")
    body.add_argument("--no-ocr", action="store_true", help="disable optional Tesseract OCR")
    body.add_argument("--list-windows", action="store_true")

    train = sub.add_parser("mm2-train", help="train/export the MM2 detector to ONNX")
    train.add_argument("--dataset", default="data/datasets/mm2/yolo/dataset.yaml")
    train.add_argument("--base-model", default="yolov8n.pt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=None, help="cpu, 0, 1, ...")
    train.add_argument("--output-dir", default="runs/mm2")
    train.add_argument("--model-output-dir", default="data/models/142823291")

    args = parser.parse_args()

    if args.command == "dashboard":
        _run_dashboard(args)
        return
    if args.command == "body":
        _run_body(args)
        return
    if args.command == "mm2-train":
        _run_mm2_train(args)
        return

    print(f"RobloxHive {__version__}")
    print("Commands:")
    print("  python -m robloxhive dashboard")
    print("  python -m robloxhive body --list-windows")
    print("  python -m robloxhive mm2-train")


if __name__ == "__main__":
    main()
