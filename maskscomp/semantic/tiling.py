from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    tile_id: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y0 + self.y1)


def generate_tiles(height: int, width: int, tile_size: int, stride: int) -> list[Tile]:
    """Deterministic top-left-origin row-major tiling with border handling."""
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")

    ys = list(range(0, max(1, height - tile_size + 1), stride))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    last_y = max(0, height - tile_size)
    last_x = max(0, width - tile_size)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)

    tiles: list[Tile] = []
    idx = 0
    for y in ys:
        for x in xs:
            x1 = min(width, x + tile_size)
            y1 = min(height, y + tile_size)
            tiles.append(Tile(tile_id=idx, x0=int(x), y0=int(y), x1=int(x1), y1=int(y1)))
            idx += 1
    return tiles
