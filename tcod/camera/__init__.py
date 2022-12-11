"""Camera helper tools for 2D tile-based projects."""
from __future__ import annotations

__version__ = "0.0.1"

from typing import Any, TypeVar, overload

try:
    from numpy.typing import NDArray
except ImportError:
    pass

_ScreenArray = TypeVar("_ScreenArray", bound="NDArray[Any]")
_WorldArray = TypeVar("_WorldArray", bound="NDArray[Any]")


def get_slice_1d(screen_width: int, world_width: int, camera_pos: int) -> tuple[slice, slice]:
    """Return a (screen_slice, world_slice) pair of slices for the given screen/world sizes with the camera position.

    Args:
        screen_width: The screen width.
        world_width: The world width.
        camera_pos: The left-most position where the camera is anchored to the world.
    """
    screen_left = max(0, -camera_pos)
    world_left = max(0, camera_pos)
    screen_width = min(screen_width - screen_left, world_width - world_left)
    return slice(screen_left, screen_left + screen_width), slice(world_left, world_left + screen_width)


@overload
def get_slices(screen: tuple[int], world: tuple[int], camera: tuple[int]) -> tuple[tuple[slice], tuple[slice]]:
    ...


@overload
def get_slices(
    screen: tuple[int, int], world: tuple[int, int], camera: tuple[int, int]
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    ...


@overload
def get_slices(
    screen: tuple[int, int, int], world: tuple[int, int, int], camera: tuple[int, int, int]
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    ...


@overload
def get_slices(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    ...


def get_slices(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return (screen_slices, world_slices) for the given parameters.

    This function takes any number of dimensions.
    The `screen`, `world`, and `camera` tuples must be the same length.

    Args:
        screen: The screen shape.
        world: The world shape.
        camera: The camera anchor position, where the camera is in the world.

    Returns:
        The (screen_slice, world_slice) slices which can be used to index arrays of the given shapes.

        Arrays indexed with these slices will always result in the same shape.
        The slices will be narrower than the screen when the camera is partially out-of-bounds.
        The slices will be zero-width if the camera is entirely out-of-bounds.
    """
    slices = (get_slice_1d(screen_, world_, camera_) for screen_, world_, camera_ in zip(screen, world, camera))
    screen_slices, world_slices = zip(*slices)
    return tuple(screen_slices), tuple(world_slices)


def get_views(
    screen: _ScreenArray, world: _WorldArray, camera_pos: tuple[int, ...]
) -> tuple[_ScreenArray, _WorldArray]:
    """Return (screen_view, world_view) for the given parameters.

    This function takes any number of dimensions.
    The `screen`, `world`, and `camera` tuples must be the same length.

    Args:
        screen: The NumPy array for the screen.
        world: The NumPy array for the world.
        camera: The camera anchor position, where the camera exists in the world.

    Returns:
        The given arrays pre-sliced into (screen_view, world_view) views.

        These will always be the same shape.
        They will be sliced into a zero-width views once the camera is far enough out-of-bounds.

        Convenient when you only have one screen and one world array to work with, otherwise you should call
        :any:`get_slices` instead.
    """
    screen_slice, world_slice = get_slices(screen.shape, world.shape, camera_pos)
    return screen[screen_slice], world[world_slice]  # type: ignore[return-value]


def clamp_camera_1d(screen_width: int, world_width: int, camera_pos: int, justify: float) -> int:
    """Clamp the camera to screen/world shapes along 1 dimension."""
    justify = min(max(0, justify), 1)
    right_bound = max(0, world_width - screen_width)
    screen_padding = max(0, screen_width - world_width)
    camera_pos = min(max(0, camera_pos), right_bound)
    camera_pos -= int(screen_padding * justify)
    return camera_pos


def clamp_camera(
    screen: tuple[int, int], world: tuple[int, int], camera: tuple[int, int], justify: tuple[float, float] = (0.5, 0.5)
) -> tuple[int, int]:
    """Clamp the camera to the screen/world shapes.  Preventing the camera from leaving the world boundary.

    Args:
        screen: The screen shape.
        world: The world shape.
        camera: The current camera position.
        justify: The justification to use when the world is smaller than the screen.
            Defaults to ``(0.5, 0.5)`` which will center the world when it is smaller than the screen.

            A value of ``(0, 0)`` will move a world smaller to the screen to inner corner.
            ``(1, 1)`` would do the same but to the opposite corner.

    Returns:
        The new camera position clamped using the given shapes and justification rules.

        Like the other functions, this camera position still assumes that the screen offset is ``(0, 0)``.
        This means that no other code changes are necessary to add or remove this clamping effect.
        This also means that changing ``justify`` also requires no external changes.
    """
    return (
        clamp_camera_1d(screen[0], world[0], camera[0], justify[0]),
        clamp_camera_1d(screen[1], world[1], camera[1], justify[1]),
    )
