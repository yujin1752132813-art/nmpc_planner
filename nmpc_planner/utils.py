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


def unwrap_to_near(angle: float, reference: float) -> float:
    """Shift angle by ±2pi so that it stays as close as possible to reference.

    This is critical for keeping yaw continuous across the ±pi boundary when:
    - the current ego yaw is wrapped to [-pi, pi]
    - the warm-start trajectory / reference yaw is unwrapped and continuous
    """
    while angle - reference > math.pi:
        angle -= 2.0 * math.pi
    while angle - reference < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))