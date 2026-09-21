from __future__ import annotations

import os
import time
from contextlib import contextmanager


class WindowInputError(RuntimeError):
    pass


class Win32MessageInput:
    """HWND-scoped message input.

    This is the least intrusive mode, but Roblox may ignore WM_KEY/WM_MOUSE
    messages because many builds read lower-level input APIs.
    """

    mode = "message"

    KEY_MAP = {
        "forward": 0x57,
        "back": 0x53,
        "left": 0x41,
        "right": 0x44,
        "jump": 0x20,
        "interact": 0x45,
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
        lparam = win32api.MAKELONG(max(0, int(x)), max(0, int(y)))
        self._post(win32con.WM_MOUSEMOVE, 0, lparam)

    def click_client(self, x: int, y: int, button: str = "left") -> None:
        try:
            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError("Win32MessageInput requires pywin32") from exc
        lparam = win32api.MAKELONG(max(0, int(x)), max(0, int(y)))
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


class ForegroundWin32Input:
    """Reliable SendInput-style controller for normal Roblox clients.

    Input is only emitted after the assigned HWND is confirmed foreground.
    The previous foreground window is restored after each short action. This
    intentionally prefers safety over accidentally sending keys to another
    Roblox/account when Windows refuses a focus change.
    """

    mode = "foreground"

    KEY_NAMES = {
        "forward": "w",
        "back": "s",
        "left": "a",
        "right": "d",
        "jump": "space",
        "interact": "e",
    }

    def __init__(self, hwnd: int, restore_focus: bool = True) -> None:
        if os.name != "nt":
            raise RuntimeError("ForegroundWin32Input is Windows-only")
        self.hwnd = hwnd
        self.restore_focus = restore_focus

    def _validate(self) -> None:
        import win32gui
        if not win32gui.IsWindow(self.hwnd):
            raise WindowInputError("BOT_WINDOW_INVALID")

    @contextmanager
    def _focused(self):
        try:
            import win32con
            import win32gui
        except ImportError as exc:
            raise RuntimeError("ForegroundWin32Input requires pywin32") from exc

        self._validate()
        previous = win32gui.GetForegroundWindow()
        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                time.sleep(0.05)
            win32gui.BringWindowToTop(self.hwnd)
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.035)
        except Exception as exc:
            raise WindowInputError(f"BOT_WINDOW_FOCUS_FAILED:{exc}") from exc

        if win32gui.GetForegroundWindow() != self.hwnd:
            raise WindowInputError("BOT_WINDOW_FOCUS_NOT_CONFIRMED")

        try:
            yield
        finally:
            if (
                self.restore_focus
                and previous
                and previous != self.hwnd
                and win32gui.IsWindow(previous)
            ):
                try:
                    win32gui.SetForegroundWindow(previous)
                except Exception:
                    pass

    def _key_name(self, name: str) -> str:
        mapped = self.KEY_NAMES.get(name)
        if mapped:
            return mapped
        if len(name) == 1:
            return name.lower()
        raise WindowInputError(f"UNKNOWN_KEY:{name}")

    def key(self, name: str, seconds: float = 0.08) -> None:
        try:
            import pydirectinput
        except ImportError as exc:
            raise RuntimeError("Foreground input requires pydirectinput") from exc
        key_name = self._key_name(name)
        duration = max(0.01, min(float(seconds), 3.0))
        with self._focused():
            pydirectinput.keyDown(key_name)
            try:
                time.sleep(duration)
            finally:
                pydirectinput.keyUp(key_name)

    def move(self, direction: str, seconds: float) -> None:
        self.key(direction, seconds)

    def interact(self, key: str = "interact") -> None:
        self.key(key, 0.06)

    def _screen_point(self, x: int, y: int) -> tuple[int, int]:
        import win32gui
        self._validate()
        sx, sy = win32gui.ClientToScreen(self.hwnd, (max(0, int(x)), max(0, int(y))))
        return int(sx), int(sy)

    def aim_client(self, x: int, y: int) -> None:
        try:
            import pydirectinput
        except ImportError as exc:
            raise RuntimeError("Foreground input requires pydirectinput") from exc
        sx, sy = self._screen_point(x, y)
        with self._focused():
            pydirectinput.moveTo(sx, sy, duration=0)

    def click_client(self, x: int, y: int, button: str = "left") -> None:
        try:
            import pydirectinput
        except ImportError as exc:
            raise RuntimeError("Foreground input requires pydirectinput") from exc
        sx, sy = self._screen_point(x, y)
        with self._focused():
            pydirectinput.moveTo(sx, sy, duration=0)
            pydirectinput.click(button="left" if button == "left" else "right")

    def release_all(self) -> None:
        try:
            import pydirectinput
        except ImportError:
            return
        try:
            with self._focused():
                for key in set(self.KEY_NAMES.values()):
                    try:
                        pydirectinput.keyUp(key)
                    except Exception:
                        pass
        except Exception:
            pass


def create_input_backend(hwnd: int, mode: str = "auto"):
    normalized = (mode or "auto").strip().lower()
    if normalized == "message":
        return Win32MessageInput(hwnd)
    if normalized in {"auto", "foreground"}:
        # Roblox frequently ignores WM_KEY messages, so v1 defaults to the
        # verified-foreground SendInput path. Users can opt into message mode.
        return ForegroundWin32Input(hwnd)
    raise ValueError(f"Unknown input mode: {mode}")
