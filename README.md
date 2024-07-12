# About

[![PyPI](https://img.shields.io/pypi/v/tcod-camera)](https://pypi.org/project/tcod-camera/)
[![PyPI - License](https://img.shields.io/pypi/l/tcod-camera)](https://github.com/HexDecimal/python-tcod-camera/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/python-tcod-camera/badge/?version=latest)](https://python-tcod-camera.readthedocs.io)
[![codecov](https://codecov.io/gh/HexDecimal/python-tcod-camera/branch/main/graph/badge.svg?token=UP161WEo0s)](https://codecov.io/gh/HexDecimal/python-tcod-camera)

This packages contains a set of tools for working with cameras which translate between world and screen coordinates.

It is intended to be used with [Python-tcod](https://github.com/libtcod/python-tcod) and [NumPy](https://numpy.org/) but requires neither.

[Additional examples can be found here.](https://github.com/HexDecimal/python-tcod-camera/tree/main/examples)

```py
# This library works with the idea that you have a world array you want projected onto a screen array.
>>> import numpy as np
>>> import tcod.camera
>>> screen = np.arange(3 * 3, dtype=int).reshape(3, 3)
>>> world = np.arange(9 * 10, dtype=int).reshape(9, 10)
>>> screen
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> world
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]])

# This example uses `ij` coordinates, but `xy` coordinates are also an option.
# The most basic example is to get the camera and use it to project the world and screen shapes.
>>> camera_ij = tcod.camera.get_camera(screen.shape, center=(2, 2))  # Get the camera position centered on (2, 2).
>>> camera_ij  # The actual camera position is always which world position to project onto screen[0, 0].
(1, 1)
>>> screen_slice, world_slice = tcod.camera.get_slices(screen.shape, world.shape, camera_ij)
>>> screen_slice
(slice(0, 3, None), slice(0, 3, None))
>>> world_slice
(slice(1, 4, None), slice(1, 4, None))
>>> screen[screen_slice] = world[world_slice]  # Project the values of screen onto the world.
>>> screen
array([[11, 12, 13],
       [21, 22, 23],
       [31, 32, 33]])

# Out-of-bounds camera coordinates result in partial views.
# Fully out-of-bounds cameras will result in zero element views.
>>> camera_ij = tcod.camera.get_camera(screen.shape, (4, 10))  # A camera position beyond the right side of the world.
>>> screen_slice, world_slice = tcod.camera.get_slices(screen.shape, world.shape, camera_ij)
>>> screen[screen_slice].shape  # Because this is partially out-of-bounds not all of the screen is in view.
(3, 1)
>>> screen_slice
(slice(0, 3, None), slice(0, 1, None))
>>> world_slice
(slice(3, 6, None), slice(9, 10, None))
>>> screen[:] = -1  # The screen will be cleared with -1, this value now means out-of-bounds.
>>> screen[screen_slice] = world[world_slice]  # The parts which do overlap will be projected.
>>> screen
array([[39, -1, -1],
       [49, -1, -1],
       [59, -1, -1]])

# By adding the shape of the world to camera functions the camera can be clamped to the bounds of the world.
# All screen indexes will be in-view as long as the screen is never larger than the world.
>>> camera_ij = tcod.camera.clamp_camera(screen.shape, world.shape, camera_ij)
>>> screen_slice, world_slice = tcod.camera.get_slices(screen.shape, world.shape, camera_ij)
>>> screen[screen_slice] = world[world_slice]
>>> screen  # The camera was moved left to fit the screen to the world.
array([[37, 38, 39],
       [47, 48, 49],
       [57, 58, 59]])

# If the world is divided into chunks then this library can be used to project each chunk onto a single screen.
# You'll have to manage your own chunks.  Possibly in a `dict[tuple[int, int], NDArray[Any]]`-like container.
>>> screen = np.zeros((10, 10), dtype=int)
>>> CHUNK_SIZE = (4, 4)
>>> for screen_slice, chunk_ij, chunk_slice in tcod.camera.get_chunked_slices(screen.shape, CHUNK_SIZE, camera=(0, 0)):
...     screen[screen_slice] = chunk_ij[0] + chunk_ij[1] * 10
...     print(f"{screen_slice=}, {chunk_ij=}, {chunk_slice=}")
screen_slice=(slice(0, 4, None), slice(0, 4, None)), chunk_ij=(0, 0), chunk_slice=(slice(0, 4, None), slice(0, 4, None))
screen_slice=(slice(0, 4, None), slice(4, 8, None)), chunk_ij=(0, 1), chunk_slice=(slice(0, 4, None), slice(0, 4, None))
screen_slice=(slice(0, 4, None), slice(8, 10, None)), chunk_ij=(0, 2), chunk_slice=(slice(0, 4, None), slice(0, 2, None))
screen_slice=(slice(4, 8, None), slice(0, 4, None)), chunk_ij=(1, 0), chunk_slice=(slice(0, 4, None), slice(0, 4, None))
screen_slice=(slice(4, 8, None), slice(4, 8, None)), chunk_ij=(1, 1), chunk_slice=(slice(0, 4, None), slice(0, 4, None))
screen_slice=(slice(4, 8, None), slice(8, 10, None)), chunk_ij=(1, 2), chunk_slice=(slice(0, 4, None), slice(0, 2, None))
screen_slice=(slice(8, 10, None), slice(0, 4, None)), chunk_ij=(2, 0), chunk_slice=(slice(0, 2, None), slice(0, 4, None))
screen_slice=(slice(8, 10, None), slice(4, 8, None)), chunk_ij=(2, 1), chunk_slice=(slice(0, 2, None), slice(0, 4, None))
screen_slice=(slice(8, 10, None), slice(8, 10, None)), chunk_ij=(2, 2), chunk_slice=(slice(0, 2, None), slice(0, 2, None))
>>> screen
array([[ 0,  0,  0,  0, 10, 10, 10, 10, 20, 20],
       [ 0,  0,  0,  0, 10, 10, 10, 10, 20, 20],
       [ 0,  0,  0,  0, 10, 10, 10, 10, 20, 20],
       [ 0,  0,  0,  0, 10, 10, 10, 10, 20, 20],
       [ 1,  1,  1,  1, 11, 11, 11, 11, 21, 21],
       [ 1,  1,  1,  1, 11, 11, 11, 11, 21, 21],
       [ 1,  1,  1,  1, 11, 11, 11, 11, 21, 21],
       [ 1,  1,  1,  1, 11, 11, 11, 11, 21, 21],
       [ 2,  2,  2,  2, 12, 12, 12, 12, 22, 22],
       [ 2,  2,  2,  2, 12, 12, 12, 12, 22, 22]])

```
