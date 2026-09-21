from __future__ import annotations

import ctypes
import os
from pathlib import Path


class NativeMath:
    """Optional Rust accelerator with pure-Python fallback."""

    def __init__(self) -> None:
        self.lib = None
        self.path: str | None = None
        candidates = []
        if os.name == "nt":
            candidates = [
                Path("native/robloxhive-native/target/release/robloxhive_native.dll"),
                Path("robloxhive_native.dll"),
            ]
        else:
            candidates = [
                Path("native/robloxhive-native/target/release/librobloxhive_native.so"),
                Path("native/robloxhive-native/target/release/librobloxhive_native.dylib"),
            ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                lib = ctypes.CDLL(str(candidate.resolve()))
                lib.rh_predict_axis.argtypes = [ctypes.c_double] * 4
                lib.rh_predict_axis.restype = ctypes.c_double
                lib.rh_iou.argtypes = [ctypes.c_double] * 8
                lib.rh_iou.restype = ctypes.c_double
                self.lib = lib
                self.path = str(candidate)
                break
            except OSError:
                continue

    @property
    def enabled(self) -> bool:
        return self.lib is not None

    def predict_axis(self, position: float, velocity: float, lead_seconds: float, max_lead: float) -> float:
        if self.lib is not None:
            return float(self.lib.rh_predict_axis(position, velocity, lead_seconds, max_lead))
        lead = max(-max_lead, min(max_lead, velocity * lead_seconds))
        return position + lead

    def iou(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        if self.lib is not None:
            return float(self.lib.rh_iou(*a, *b))
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
        return inter / union if union > 0 else 0.0


native_math = NativeMath()
