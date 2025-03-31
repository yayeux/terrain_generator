def fade(t: float) -> float:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)