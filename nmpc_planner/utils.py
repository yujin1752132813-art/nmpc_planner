from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def unwrap_sequence(seq):
    if not seq:
        return []
    out = [seq[0]]
    for val in seq[1:]:
        prev = out[-1]
        delta = wrap_angle(val - prev)
        out.append(prev + delta)
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))