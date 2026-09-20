from __future__ import annotations

import os

import numpy as np


class WindowCaptureError(RuntimeError):
    pass


class Win32WindowCapture:
    """Capture a specific HWND with PrintWindow.

    PrintWindow is intentionally tied to the assigned bot HWND. Some GPU-backed
    Roblox versions may return a black frame; callers must detect that and fail
    safely instead of falling back to desktop capture (which could observe the
    human player's window).
    """

    def __init__(self, hwnd: int) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32WindowCapture is Windows-only")
        self.hwnd = hwnd

    def capture(self) -> np.ndarray:
        try:
            import win32con
            import win32gui
            import win32ui
        except ImportError as exc:
            raise RuntimeError("Win32WindowCapture requires pywin32") from exc

        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width, height = right - left, bottom - top
        if width <= 1 or height <= 1:
            raise WindowCaptureError("BOT_WINDOW_HAS_NO_CLIENT_AREA")

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        src_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = src_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(src_dc, width, height)
        mem_dc.SelectObject(bitmap)

        try:
            ok = win32gui.PrintWindow(self.hwnd, mem_dc.GetSafeHdc(), 3)
            info = bitmap.GetInfo()
            bits = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bits, dtype=np.uint8)
            image.shape = (info["bmHeight"], info["bmWidth"], 4)
            frame = image[:, :, :3].copy()
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            mem_dc.DeleteDC()
            src_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        if not ok or frame.size == 0:
            raise WindowCaptureError("PRINTWINDOW_FAILED")
        if float(frame.std()) < 1.0:
            raise WindowCaptureError("CAPTURE_LOOKS_BLANK")
        return frame
