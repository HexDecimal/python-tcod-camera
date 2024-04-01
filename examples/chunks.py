#!/usr/bin/env python
"""Example script for infinite-world chunked maps."""

from __future__ import annotations

import random
from typing import Any, DefaultDict, Tuple

import attrs
import numpy as np
from numpy.typing import NDArray

import tcod.camera
import tcod.console
import tcod.context
import tcod.event


@attrs.define
class Thing:
    """A distinct object."""

    x: int
    y: int
    ch: int
    fg: tuple[int, int, int] = (255, 255, 255)


FLOOR_GRAPHICS: NDArray[np.int32] = np.array([ord(ch) for ch in "    ,.'`"], dtype=np.int32)


MOVE_KEYS = {
    # Arrow keys.
    tcod.event.Scancode.UP: (0, -1),
    tcod.event.Scancode.DOWN: (0, 1),
    tcod.event.Scancode.LEFT: (-1, 0),
    tcod.event.Scancode.RIGHT: (1, 0),
    tcod.event.Scancode.HOME: (-1, -1),
    tcod.event.Scancode.END: (-1, 1),
    tcod.event.Scancode.PAGEUP: (1, -1),
    tcod.event.Scancode.PAGEDOWN: (1, 1),
    # Numpad keys.
    tcod.event.Scancode.KP_1: (-1, 1),
    tcod.event.Scancode.KP_2: (0, 1),
    tcod.event.Scancode.KP_3: (1, 1),
    tcod.event.Scancode.KP_4: (-1, 0),
    tcod.event.Scancode.KP_6: (1, 0),
    tcod.event.Scancode.KP_7: (-1, -1),
    tcod.event.Scancode.KP_8: (0, -1),
    tcod.event.Scancode.KP_9: (1, -1),
    # Vi keys.
    tcod.event.Scancode.H: (-1, 0),
    tcod.event.Scancode.J: (0, 1),
    tcod.event.Scancode.K: (0, -1),
    tcod.event.Scancode.L: (1, 0),
    tcod.event.Scancode.Y: (-1, -1),
    tcod.event.Scancode.U: (1, -1),
    tcod.event.Scancode.B: (-1, 1),
    tcod.event.Scancode.N: (1, 1),
    # WASD/WAXD keys.
    tcod.event.Scancode.Z: (-1, 1),
    tcod.event.Scancode.X: (0, 1),
    tcod.event.Scancode.C: (1, 1),
    tcod.event.Scancode.A: (-1, 0),
    tcod.event.Scancode.S: (0, 1),
    tcod.event.Scancode.D: (1, 0),
    tcod.event.Scancode.Q: (-1, -1),
    tcod.event.Scancode.W: (0, -1),
    tcod.event.Scancode.E: (1, -1),
}
CHUNK_SHAPE = 12, 16  # Chunk (height, width)


def new_chunk(i: int, j: int) -> NDArray[Any]:
    """Return a new map chunk, annotated with its position."""
    chunk = np.zeros(CHUNK_SHAPE, dtype=tcod.console.rgb_graphic)
    chunk["ch"] = FLOOR_GRAPHICS[np.random.randint(FLOOR_GRAPHICS.size, size=CHUNK_SHAPE)]
    chunk["fg"] = 0x88, 0x88, 0x88
    chunk["bg"] = random.randint(0, 0x40), random.randint(0, 0x40), random.randint(0, 0x40)

    string = [ord(c) for c in f"({j} {i}) "]
    str_size = min(len(string), chunk.shape[1])
    chunk["ch"][0, :str_size] = string[:str_size]
    return chunk


class ChunkedWorld(DefaultDict[Tuple[int, ...], NDArray[Any]]):
    """A collection of chunks, chunks are generated on demand as they are indexed."""

    def __missing__(self, __key: tuple[int, ...]) -> NDArray[Any]:
        """Generate chunks on-demand."""
        self[__key] = value = new_chunk(*__key)
        return value


def main() -> None:
    """Begin a demo where the player can explore an infinite world."""
    context = tcod.context.new()
    player = Thing(CHUNK_SHAPE[1] // 2, CHUNK_SHAPE[0] // 2, ord("@"))
    things = [player]

    chunks = ChunkedWorld()

    while True:
        console = context.new_console(10, 10)

        camera_ij = tcod.camera.get_camera(console.rgb.shape, (player.y, player.x))

        for screen_slice, chunk_id, chunk_slice in tcod.camera.get_chunked_slices(
            console.rgb.shape, CHUNK_SHAPE, camera_ij
        ):
            console.rgb[screen_slice] = chunks[chunk_id][chunk_slice]

        console_ch_fg = console.rgb[["ch", "fg"]]
        for thing in things:
            y = thing.y - camera_ij[0]
            x = thing.x - camera_ij[1]
            if not (0 <= x < console.width and 0 <= y < console.height):
                continue
            console_ch_fg[y, x] = thing.ch, thing.fg

        console.print(0, 0, f"Player pos: {player.x},{player.y}", fg=(255, 255, 255))
        console.print(0, 1, f"Camera pos: {camera_ij[1]},{camera_ij[0]}", fg=(255, 255, 255))

        context.present(console, keep_aspect=True, integer_scaling=True)

        for event in tcod.event.wait():
            if isinstance(event, tcod.event.Quit):
                raise SystemExit()
            if isinstance(event, tcod.event.KeyDown) and event.scancode in MOVE_KEYS:
                dx, dy = MOVE_KEYS[event.scancode]
                player.x += dx
                player.y += dy


if __name__ == "__main__":
    main()
