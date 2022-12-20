#!/sr/bin/env python
from __future__ import annotations

import random
from typing import Any, Tuple

import attrs
import numpy as np
from numpy.typing import NDArray

import tcod
import tcod.camera


@attrs.define
class Thing:
    x: int
    y: int
    ch: int
    fg: Tuple[int, int, int] = (255, 255, 255)


FLOOR_GRAPHICS = np.array([ord(ch) for ch in " ,.'`"], dtype=np.int32)


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
}
MAP_WIDTH, MAP_HEIGHT = 50, 50


def main() -> None:
    context = tcod.context.new()
    player = Thing(MAP_WIDTH // 2, MAP_HEIGHT // 2, ord("@"))
    things = [
        *(
            Thing(random.randint(0, MAP_WIDTH - 1), random.randint(0, MAP_HEIGHT - 1), ord("%"), (255, 255, 0))
            for _ in range(10)
        ),
        player,
    ]

    world: NDArray[Any] = FLOOR_GRAPHICS[np.random.randint(FLOOR_GRAPHICS.size, size=(MAP_HEIGHT, MAP_WIDTH))]
    clamp = False

    while True:
        console = context.new_console(10, 10)

        camera_ij = tcod.camera.get_camera(console.rgb.shape, (player.y, player.x))
        if clamp:
            camera_ij = tcod.camera.clamp_camera(console.rgb.shape, world.shape, camera_ij, (0.5, 0.5))

        screen_view, world_view = tcod.camera.get_views(console.rgb, world, camera_ij)
        screen_view["ch"] = world_view
        screen_view["fg"] = (0x88, 0x88, 0x88)
        screen_view["bg"] = (0x8, 0x8, 0x8)

        console_ch_fg = console.rgb[["ch", "fg"]]
        for thing in things:
            y = thing.y - camera_ij[0]
            x = thing.x - camera_ij[1]
            if not (0 <= x < console.width and 0 <= y < console.height):
                continue
            console_ch_fg[y, x] = thing.ch, thing.fg

        console.print(0, 0, f"{clamp=} (press TAB to toggle)", fg=(255, 255, 255))
        console.print(0, 1, f"Player pos: {player.x},{player.y}", fg=(255, 255, 255))
        console.print(0, 2, f"Camera pos: {camera_ij[1]},{camera_ij[0]}", fg=(255, 255, 255))

        context.present(console, keep_aspect=True, integer_scaling=True)

        for event in tcod.event.wait():
            if isinstance(event, tcod.event.Quit):
                raise SystemExit()
            if isinstance(event, tcod.event.KeyDown):
                if event.scancode in MOVE_KEYS:
                    dx, dy = MOVE_KEYS[event.scancode]
                    player.x += dx
                    player.y += dy
                if event.sym == tcod.event.KeySym.TAB:
                    clamp = not clamp


if __name__ == "__main__":
    main()
