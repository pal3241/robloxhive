from __future__ import annotations

import os
from robloxhive.body.instances import RobloxInstance


def discover_roblox_windows() -> list[RobloxInstance]:
    """Discover top-level Roblox windows on Windows and bind each to PID/HWND.

    This function performs discovery only. It never decides which window belongs
    to the human player or to an agent.
    """

    if os.name != "nt":
        return []

    try:
        import psutil
        import win32gui
        import win32process
    except ImportError as exc:
        raise RuntimeError(
            "Windows discovery requires psutil and pywin32. "
            "Install RobloxHive with the windows extra."
        ) from exc

    instances: list[RobloxInstance] = []

    def callback(hwnd: int, _extra: object) -> bool:
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return True

        # EnumWindows can surface IME/helper windows owned by RobloxPlayerBeta.
        # They share the Roblox PID but are not controllable game windows.
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True
        if title.lower() in {"default ime", "msctfime ui"}:
            return True
        if win32gui.GetParent(hwnd):
            return True

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            process_name = process.name().lower()
        except (psutil.Error, OSError):
            return True

        if "roblox" not in process_name:
            return True

        # The actual Roblox render window has a normal non-empty client area.
        # Reject zero/tiny helper surfaces defensively.
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
        except OSError:
            return True
        if (right - left) < 160 or (bottom - top) < 120:
            return True

        instances.append(
            RobloxInstance(
                pid=pid,
                hwnd=hwnd,
                title=title,
                alive=process.is_running(),
            )
        )
        return True

    win32gui.EnumWindows(callback, None)
    unique: dict[tuple[int, int], RobloxInstance] = {}
    for instance in instances:
        unique[(instance.pid, instance.hwnd)] = instance
    return list(unique.values())
