"""Tests for tcod.camera."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import tcod.camera

# ruff: noqa: D103


def test_views() -> None:
    screen: NDArray[np.int32] = np.zeros((2, 3), dtype=np.int32)
    world: NDArray[np.int32] = np.arange(4 * 5, dtype=np.int32).reshape((4, 5))

    screen_view, world_view = tcod.camera.get_views(screen, world, (-1000, -1000))
    assert screen_view.shape == world_view.shape
    assert screen_view.size == 0
    assert world_view.size == 0

    screen_view, world_view = tcod.camera.get_views(screen, world, (1000, 1000))
    assert screen_view.size == 0
    assert world_view.size == 0

    screen_view, world_view = tcod.camera.get_views(screen, world, (1, 1))
    assert screen_view.shape == world_view.shape
    assert world_view.tolist() == [[6, 7, 8], [11, 12, 13]]

    for i in range(-10, 10):
        for j in range(-10, 10):
            screen_view, world_view = tcod.camera.get_views(screen, world, (i, j))
            screen_view[:] = world_view[:]
            assert screen_view.shape == world_view.shape, f"{i=},{j=}, {screen_view=}, {world_view=}"


def test_get_camera() -> None:
    assert tcod.camera.get_camera((3, 2), (5, 5)) == (5 - 3 // 2, 5 - 2 // 2)
    assert tcod.camera.get_camera((3, 2), (5, 5), ((1, 1), 0)) == (0, 0)
    assert tcod.camera.get_camera((3, 2), (5, 5), ((1, 1), 1)) == (-2, -1)

    assert [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3] == [
        tcod.camera.get_camera((2,), (i,), ((5,), 0))[0] for i in range(-5, 10)
    ]
    assert [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] == [
        tcod.camera.get_camera((5,), (i,), ((2,), 0))[0] for i in range(-5, 10)
    ]
    assert [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3] == [
        tcod.camera.get_camera((5,), (i,), ((2,), 1))[0] for i in range(-5, 10)
    ]
