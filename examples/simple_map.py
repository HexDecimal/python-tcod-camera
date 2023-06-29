#!/usr/bin/env python
"""Example script for a camera on a single map."""
from __future__ import annotations

import random
from typing import Any

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
MAP_WIDTH, MAP_HEIGHT = 50, 50


def main() -> None:  # noqa: C901, PLR0912
    """Begin a demo where a player can control a character and the camera."""
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
    clamp = False  # If True then the camera will be bound to the world edges.
    cursor_screen_xy: None | tuple[int, int] = None  # Cursor position in screen space.
    cursor_world_xy: None | tuple[int, int] = None  # Cursor position in world space.
    offset_xy = (0, 0)  # Camera offset from the player position.

    while True:
        console = context.new_console(10, 10)

        camera_ij = tcod.camera.get_camera(console.rgb.shape, (player.y + offset_xy[1], player.x + offset_xy[0]))
        if clamp:
            camera_ij = tcod.camera.clamp_camera(console.rgb.shape, world.shape, camera_ij, (0.5, 0.5))

        # Any movements of the camera have to be reflected in the cursor world position.
        cursor_world_xy = None
        if cursor_screen_xy:
            cursor_world_xy = cursor_screen_xy[0] + camera_ij[1], cursor_screen_xy[1] + camera_ij[0]

        screen_view, world_view = tcod.camera.get_views(console.rgb, world, camera_ij)
        screen_view["ch"] = world_view
        screen_view["fg"] = (0x88, 0x88, 0x88)
        screen_view["bg"] = (0x8, 0x8, 0x8)

        console_ch_fg = console.rgb[["ch", "fg"]]
        for thing in things:
            y = thing.y - camera_ij[0]
            x = thing.x - camera_ij[1]
            if not (0 <= x < console.width and 0 <= y < console.height):
                continue  # Cull out-of-bounds objects.
            console_ch_fg[y, x] = thing.ch, thing.fg

        console.print_box(
            0,
            0,
            console.width,
            0,
            f"{clamp=} (press TAB to toggle)"
            f"\nOffset pos: {offset_xy} (right click and drag to adjust)"
            f"\nPlayer pos: {player.x, player.y}"
            f"\nCamera pos: {camera_ij[1], camera_ij[0]}"
            f"\nCursor screen pos: {cursor_screen_xy}"
            f"\nCursor world pos: {cursor_world_xy}"
            # Find world space objects under the cursor.
            f"\nCursor objs: {[thing for thing in things if (thing.x, thing.y) == cursor_world_xy]}",
            fg=(255, 255, 255),
        )

        # Highlight the tile under the mouse.
        if cursor_screen_xy and 0 <= cursor_screen_xy[0] < console.width and 0 <= cursor_screen_xy[1] < console.height:
            console.rgb[["fg", "bg"]][cursor_screen_xy[1], cursor_screen_xy[0]] = (0, 0, 0), (255, 255, 255)

        context.present(console, keep_aspect=True, integer_scaling=True)

        for event in tcod.event.wait():
            context.convert_event(event)
            if isinstance(event, tcod.event.Quit):
                raise SystemExit()
            if isinstance(event, tcod.event.KeyDown):
                if event.scancode in MOVE_KEYS:
                    dx, dy = MOVE_KEYS[event.scancode]
                    player.x += dx
                    player.y += dy
                if event.sym == tcod.event.KeySym.TAB:
                    clamp = not clamp
            elif isinstance(event, tcod.event.MouseMotion):
                if event.state & tcod.event.BUTTON_RMASK:  # Move offset by tile motion.
                    offset_xy = offset_xy[0] - event.tile_motion.x, offset_xy[1] - event.tile_motion.y
                cursor_screen_xy = event.tile.x, event.tile.y  # Save screen cursor position.
            elif isinstance(event, tcod.event.WindowEvent) and event.type == "WINDOWLEAVE":  # type: ignore[comparison-overlap]
                cursor_screen_xy = cursor_world_xy = None  # Reset cursor when the mouse leaves the window.


if __name__ == "__main__":
    main()
