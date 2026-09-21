from __future__ import annotations

import os
import time


class WindowInputError(RuntimeError):
    pass


class Win32MessageInput:
    """Best-effort HWND-scoped input without global cursor/keyboard takeover.

    Roblox may ignore WM_KEY* messages in some builds because games often use
    lower-level input APIs. This backend therefore never claims delivery; skill
    verification must confirm movement/action from perception.
    """

    KEY_MAP = {
        "forward": 0x57,  # W
        "back": 0x53,     # S
        "left": 0x41,     # A
        "right": 0x44,    # D
        "jump": 0x20,     # SPACE
        "interact": 0x45, # E
    }

    def __init__(self, hwnd: int) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32MessageInput is Windows-only")
        self.hwnd = hwnd

    def _post(self, msg: int, vk: int, lparam: int = 0) -> None:
        try:
            import win32gui
        except ImportError as exc:
            raise RuntimeError("Win32MessageInput requires pywin32") from exc
        if not win32gui.IsWindow(self.hwnd):
            raise WindowInputError("BOT_WINDOW_INVALID")
        win32gui.PostMessage(self.hwnd, msg, vk, lparam)

    def key(self, name: str, seconds: float = 0.08) -> None:
        try:
            import win32con
        except ImportError as exc:
            raise RuntimeError("Win32MessageInput requires pywin32") from exc
        vk = self.KEY_MAP.get(name)
        if vk is None:
            if len(name) == 1:
                vk = ord(name.upper())
            else:
                raise WindowInputError(f"UNKNOWN_KEY:{name}")
        self._post(win32con.WM_KEYDOWN, vk, 0)
        time.sleep(max(0.01, min(seconds, 3.0)))
        self._post(win32con.WM_KEYUP, vk, 0)

    def move(self, direction: str, seconds: float) -> None:
        self.key(direction, seconds)

    def interact(self, key: str = "interact") -> None:
        self.key(key, 0.06)

    def aim_client(self, x: int, y: int) -> None:
        try:
            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError("Win32MessageInput requires pywin32") from exc
        lparam = win32api.MAKELONG(max(0, x), max(0, y))
        self._post(win32con.WM_MOUSEMOVE, 0, lparam)

    def click_client(self, x: int, y: int, button: str = "left") -> None:
        try:
            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError("Win32MessageInput requires pywin32") from exc
        lparam = win32api.MAKELONG(max(0, x), max(0, y))
        if button == "left":
            self._post(win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
            self._post(win32con.WM_LBUTTONUP, 0, lparam)
        else:
            self._post(win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
            self._post(win32con.WM_RBUTTONUP, 0, lparam)

    def release_all(self) -> None:
        try:
            import win32con
        except ImportError:
            return
        for vk in self.KEY_MAP.values():
            try:
                self._post(win32con.WM_KEYUP, vk, 0)
            except Exception:
                pass
