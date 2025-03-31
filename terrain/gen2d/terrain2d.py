from dataclasses import dataclass, field
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from terrain.core.utils import fade, lerp

@dataclass
class TerrainGenerator2D:
    width: int
    height: int
    scale: float
    seed: int
    octaves: int = 6
    persistence: float = 0.45
    lacunarity: float = 1.75

    heightmap: np.ndarray = field(init=False)
    _gradient_vectors: list = field(init=False)

    def __post_init__(self):
        random.seed(self.seed)

        self._gradient_vectors = [
            (math.cos(angle), math.sin(angle))
            for angle in [random.uniform(0, 2*math.pi) for _ in range(256)]
        ]

        self.heightmap = np.zeros((self.height, self.width))


    def _get_gradient_vector(self, x: int, y: int) -> tuple[float, float]:
        index = (x * 17 + y * 23) % len(self._gradient_vectors)

        return self._gradient_vectors[index]


    def _dot_product(self, g: tuple[float, float], d: tuple[float, float]) -> float:
        return g[0] * d[0] + g[1] * d[1]


    def _perlin(self, x: float, y: float) -> float:
        """Find the grid cell"""
        x0 = math.floor(x)
        x1 = x0 + 1
        y0 = math.floor(y)
        y1 = y0 + 1

        """Get the offset vectors"""
        dx = x - x0
        dy = y - y0

        """Get gradients"""
        n00 = self._dot_product(self._get_gradient_vector(x0, y0), (dx, dy))
        n10 = self._dot_product(self._get_gradient_vector(x1, y0), (dx - 1, dy))
        n01 = self._dot_product(self._get_gradient_vector(x0, y1), (dx, dy - 1))
        n11 = self._dot_product(self._get_gradient_vector(x1, y1), (dx - 1, dy - 1))

        """Smoothen the interpolation"""
        u = fade(dx)
        v = fade(dy)

        """Interpolate, then blend rows"""
        ix0 = lerp(n00, n10, u)
        ix1 = lerp(n01, n11, u)
        value = lerp(ix0, ix1, v)

        return (value + 1) / 2


    def _fbm(self, x: float, y: float) -> float:
        total = 0.0
        frequency = 1.0 # more detailed terrain
        amplitude = 1.0 # intensity/harshness of the terrain
        max_value = 0.0

        for _ in range(self.octaves):
            total += self._perlin(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return total / max_value


    def generate_heightmap(self) -> None:
        for i in range(self.height):
            for j in range(self.width):
                x = j * self.scale
                y = i * self.scale
                self.heightmap[i][j] = self._fbm(x, y)


    def save_image(self, filename: str = "fbm_heightmap.png") -> None:
        plt.figure(figsize=(8, 8))
        plt.imshow(self.heightmap, cmap="terrain")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()



if __name__ == "__main__":
    terrain = TerrainGenerator2D(width=100, height=100, scale=0.04, seed=72)
    terrain.generate_heightmap()
    terrain.save_image()