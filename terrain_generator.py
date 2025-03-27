import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math


class TerrainGenerator:
    def __init__(self, width: int, height: int, scale: float, seed: int, octaves: int = 6, persistence: float = 0.5, lacunarity: float = 2.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.heightmap: np.ndarray = np.zeros((height, width))


    def _hash_seed(self, x: int, y: int) -> int:
        key = f"{x},{y},{self.seed}"
        hash_object = hashlib.sha256(key.encode())
        hex_digest = hash_object.hexdigest()

        return int(hex_digest[12:17], 16)


    def _normalize_to_angle(self, x: int, y: int) -> float:
        max_hex_value = 16**5 - 1 # 5 characters, 4 bits for a digit (0-15 in decimal, #FFFFF in hex)
        hashed_seed = self._hash_seed(x, y) 

        return (hashed_seed / max_hex_value) * 2 * math.pi # value between 0 and 2 pi [rad]


    def _get_gradient_vector(self, x: int, y: int) -> tuple[float, float]:
        angle = self._normalize_to_angle(x, y)

        return (math.cos(angle), math.sin(angle))


    def _dot_product(self, g: tuple[float, float], d: tuple[float, float]) -> float:
        return g[0] * d[0] + g[1] * d[1]


    def _fade(self, t: float) -> float:
        return 6 * t**5 - 15 * t**4 + 10 * t**3 # bless you Ken Perlin


    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a) # praise the math gods


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
        u = self._fade(dx)
        v = self._fade(dy)

        """Interpolate, then blend rows"""
        ix0 = self._lerp(n00, n10, u)
        ix1 = self._lerp(n01, n11, u)
        value = self._lerp(ix0, ix1, v)

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
        plt.colorbar(label="Elevation")
        plt.title("FBM and Perlin Noise Heatmap")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()



if __name__ == "__main__":
    terrain = TerrainGenerator(width=100, height=100, scale=0.02, seed=42)
    terrain.generate_heightmap()
    terrain.save_image()