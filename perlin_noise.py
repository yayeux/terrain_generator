import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math

def hash_seed(x: int, y: int, seed: int) -> int:
    key = f"{x},{y},{seed}"
    hash_object = hashlib.sha256(key.encode())
    hex_digest = hash_object.hexdigest()
    sliced_seed = int(hex_digest[12:17], 16)

    return sliced_seed


def normalize_to_angle(x: int, y: int, seed: int) -> float:
    max_hex_value = 16**5 - 1 # 5 characters, 4 bits for a digit (0-15 in decimal, #FFFFF in hex)
    hashed_seed = hash_seed(x, y, seed)

    angle = (hashed_seed / max_hex_value) * 2 * math.pi # value between 0 and 2 pi [rad]

    return angle


def get_gradient_vector(x: int, y: int, seed: int) -> tuple[float, float]:
    angle = normalize_to_angle(x, y, seed)

    gx = math.cos(angle)
    gy = math.sin(angle)

    return (gx, gy)


def dot_product(g: tuple[float, float], d: tuple[float, float]) -> float:
    return g[0] * d[0] + g[1] * d[1]
 

def fade(t: float) -> float:
    return 6 * t**5 - 15 * t**4 + 10 * t**3 # bless you Ken Perlin
 

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a) # praise the math gods


def perlin(x: int, y: int, seed: int):
    """Find the grid cell"""
    x0 = math.floor(x)
    x1 = x0 + 1
    y0 = math.floor(y)
    y1 = y0 + 1

    """Get offset vectors"""
    dx = x - x0
    dy = y - y0

    """Get gradients """
    n00 = dot_product(get_gradient_vector(x0, y0, seed), (dx, dy))
    n10 = dot_product(get_gradient_vector(x1, y0, seed), (dx - 1, dy))
    n01 = dot_product(get_gradient_vector(x0, y1, seed), (dx, dy - 1))
    n11 = dot_product(get_gradient_vector(x1, y1, seed), (dx - 1, dy - 1))

    """Smoothen the interpolation"""
    u = fade(dx)
    v = fade(dy)

    """Interpolate, then blend rows"""
    ix0 = lerp(n00, n10, u)  # top row
    ix1 = lerp(n01, n11, u)  # bottom row
    value = lerp(ix0, ix1, v)

    return (value + 1) / 2


def fbm(x: float, y: float, seed: int, octaves: int = 6, persistence: float = 0.5, lacunarity: float = 2.0) -> float:
    total = 0.0
    frequency = 1.0 # more detailed terrain
    amplitude = 1.0 # intensity/harshness of the terrain
    max_value = 0.0  

    for _ in range(octaves):
        total += perlin(x * frequency, y * frequency, seed) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value
  


if __name__ == "__main__":
    width, height = 100, 100
    scale = 0.02
    seed = 42

    heightmap = np.zeros((height, width))

    for i in range(width):
        for j in range(height):
            x = i * scale
            y = j * scale
            heightmap[(i, j)] = fbm(x, y, seed)

    plt.figure(figsize=(8, 8))
    plt.imshow(heightmap, cmap="terrain")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("fbm_heightmap.png", dpi=300)
    plt.close()
