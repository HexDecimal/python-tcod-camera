"""Camera helper tools for 2D tile-based projects."""

from __future__ import annotations

__version__ = "1.0.1"

import itertools
from typing import TYPE_CHECKING, Any, Iterator, TypeVar, overload

if TYPE_CHECKING:
    from numpy.typing import NDArray


_ScreenArray = TypeVar("_ScreenArray", bound="NDArray[Any]")
_WorldArray = TypeVar("_WorldArray", bound="NDArray[Any]")


def _get_slices_1d(screen_width: int, world_width: int, camera_pos: int) -> tuple[slice, slice]:
    """Return a (screen_slice, world_slice) pair of slices for the given screen/world sizes with the camera position.

    Args:
        screen_width: The screen width.
        world_width: The world width.
        camera_pos: The left-most position where the camera is anchored to the world.
    """
    screen_left = max(0, -camera_pos)  # Screen moves right as camera moves into negatives.
    world_left = max(0, camera_pos)  # World is clamped to zero as camera moves into negatives.
    world_left = min(world_left, world_width)  # Handle camera far right out of bounds.
    # Handle screen and world size variations, and all other out-of-bounds cases.
    screen_width = max(0, min(screen_width - screen_left, world_width - world_left))
    return slice(screen_left, screen_left + screen_width), slice(world_left, world_left + screen_width)


@overload
def get_slices(screen: tuple[int], world: tuple[int], camera: tuple[int]) -> tuple[tuple[slice], tuple[slice]]: ...
@overload
def get_slices(
    screen: tuple[int, int], world: tuple[int, int], camera: tuple[int, int]
) -> tuple[tuple[slice, slice], tuple[slice, slice]]: ...
@overload
def get_slices(
    screen: tuple[int, int, int], world: tuple[int, int, int], camera: tuple[int, int, int]
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]: ...
@overload
def get_slices(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]: ...


def get_slices(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return (screen_slices, world_slices) for the given parameters.

    This function takes any number of dimensions.
    The `screen`, `world`, and `camera` tuples must be the same length.

    Args:
        screen: The :term:`screen` shape.
        world: The :term:`world` shape.
        camera: The :term:`camera` position.

    Returns:
        The (screen_slice, world_slice) slices which can be used to index arrays of the given shapes.

        Arrays indexed with these slices will always result in the same shape.
        The slices will be narrower than the screen when the camera is partially out-of-bounds.
        The slices will be zero-width if the camera is entirely out-of-bounds.

    Example::

        console: tcod.console.Console  # Libtcod console, C order.
        player_ij: tuple[int, int]  # Player (y, x) position.
        world: NDArray[Any]  # Array created with `dtype=tcod.console.rgb_graphic`, C order.

        console.clear()  # Clear the console in case any areas are not covered by tcod.camera.get_slices.

        # Get the camera position centered on the player.
        camera_ij = tcod.camera.get_camera(console.rgb.shape, player_ij)

        # Get the screen/world slices at the camera position.
        screen_slice, world_slice = tcod.camera.get_slices(console.rgb, world, camera_ij)
        console.rgb[screen_slice] = world[world_slice]  # Render world graphics.

        # Render the player.
        player_screen_y, player_screen_x = player_ij[0] - camera_ij[0], player_ij[1] - camera_ij[1]
        console.print(player_screen_x, player_screen_y, "@")
    """
    assert len(screen) == len(world) == len(camera)
    slices = (_get_slices_1d(screen_, world_, camera_) for screen_, world_, camera_ in zip(screen, world, camera))
    screen_slices, world_slices = zip(*slices)
    return tuple(screen_slices), tuple(world_slices)


def get_views(screen: _ScreenArray, world: _WorldArray, camera: tuple[int, ...]) -> tuple[_ScreenArray, _WorldArray]:
    """Return (screen_view, world_view) for the given parameters.

    This function takes any number of dimensions.
    The `screen`, `world`, and `camera` tuples must be the same length.

    Args:
        screen: The NumPy array for the :term:`screen`.
        world: The NumPy array for the :term:`world`.
        camera: The :term:`camera` position.

    Returns:
        The given arrays pre-sliced into (screen_view, world_view) views.

        These will always be the same shape.
        They will be sliced into a zero-width views once the camera is far enough out-of-bounds.

        Convenient when you only have one screen and one world array to work with, otherwise you should call
        :any:`get_slices` instead.
    """
    screen_slice, world_slice = get_slices(screen.shape, world.shape, camera)
    return screen[screen_slice], world[world_slice]  # type: ignore[return-value]


def _clamp_camera_1d(screen_width: int, world_width: int, camera_pos: int, justify: float) -> int:
    """Clamp the camera to screen/world shapes along 1 dimension."""
    justify = min(max(0, justify), 1)
    right_bound = max(0, world_width - screen_width)
    screen_padding = max(0, screen_width - world_width)
    camera_pos = min(max(0, camera_pos), right_bound)
    camera_pos -= int(screen_padding * justify)
    return camera_pos


@overload
def clamp_camera(
    screen: tuple[int], world: tuple[int], camera: tuple[int], justify: float | tuple[float] = 0.5
) -> tuple[int]: ...
@overload
def clamp_camera(
    screen: tuple[int, int], world: tuple[int, int], camera: tuple[int, int], justify: float | tuple[float, float] = 0.5
) -> tuple[int, int]: ...
@overload
def clamp_camera(
    screen: tuple[int, int, int],
    world: tuple[int, int, int],
    camera: tuple[int, int, int],
    justify: float | tuple[float, float, float] = 0.5,
) -> tuple[int, int, int]: ...
@overload
def clamp_camera(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...], justify: float | tuple[float, ...] = 0.5
) -> tuple[int, ...]: ...


def clamp_camera(
    screen: tuple[int, ...], world: tuple[int, ...], camera: tuple[int, ...], justify: float | tuple[float, ...] = 0.5
) -> tuple[int, ...]:
    """Clamp the camera to the screen/world shapes.  Preventing the camera from leaving the world boundary.

    Args:
        screen: The :term:`screen` shape.
        world: The :term:`world` shape.
        camera: The current :term:`camera` position.
        justify: The justification to use when the world is smaller than the screen.
            Defaults to ``0.5`` which will center the world when it is smaller than the screen.

            A value of zero will move a world smaller to the screen to inner corner.
            One would do the same but to the opposite corner.
            You may also give a tuple with a value for each axis.

    Returns:
        The new :term:`camera` position clamped using the given shapes and justification rules.

        Like the other functions, this camera position still assumes that the screen offset is ``(0, 0)``.
        This means that no other code changes are necessary to add or remove this clamping effect.
        This also means that changing ``justify`` also requires no external changes.
    """
    assert len(screen) == len(world) == len(camera)
    if not isinstance(justify, tuple):
        justify = (justify,) * len(camera)
    assert len(camera) == len(justify)
    return tuple(
        _clamp_camera_1d(screen_, world_, camera_, justify_)
        for screen_, world_, camera_, justify_ in zip(screen, world, camera, justify)
    )


def _get_chunked_slices_1d(screen_width: int, chunk_size: int, camera_pos: int) -> Iterator[tuple[slice, int, slice]]:
    """Iterate chucked slices along an axis.

    >>> list(_get_chunked_slices_1d(20, 10, 0))
    [(slice(0, 10, None), 0, slice(0, 10, None)), (slice(10, 20, None), 1, slice(0, 10, None))]
    >>> list(_get_chunked_slices_1d(20, 10, -5))
    [(slice(0, 5, None), -1, slice(5, 10, None)), (slice(5, 15, None), 0, slice(0, 10, None)), (slice(15, 20, None), 1, slice(0, 5, None))]
    """
    assert chunk_size > 0
    screen_left = 0
    segment_id, chunk_left = divmod(camera_pos, chunk_size)
    while screen_left < screen_width:
        # Get width until the end of the chunk or the end of the screen.
        chunk_width = min(chunk_size - chunk_left, screen_width - screen_left)
        yield slice(screen_left, screen_left + chunk_width), segment_id, slice(chunk_left, chunk_left + chunk_width)
        chunk_left = 0  # All chunks after the first start at zero.
        screen_left += chunk_width
        segment_id += 1


@overload
def get_chunked_slices(
    screen: tuple[int], chunk_shape: tuple[int], camera: tuple[int]
) -> Iterator[tuple[tuple[slice], tuple[int], tuple[slice]]]: ...
@overload
def get_chunked_slices(
    screen: tuple[int, int], chunk_shape: tuple[int, int], camera: tuple[int, int]
) -> Iterator[tuple[tuple[slice, slice], tuple[int, int], tuple[slice, slice]]]: ...
@overload
def get_chunked_slices(
    screen: tuple[int, int, int], chunk_shape: tuple[int, int, int], camera: tuple[int, int, int]
) -> Iterator[tuple[tuple[slice, slice, slice], tuple[int, int, int], tuple[slice, slice, slice]]]: ...
@overload
def get_chunked_slices(
    screen: tuple[int, ...], chunk_shape: tuple[int, ...], camera: tuple[int, ...]
) -> Iterator[tuple[tuple[slice, ...], tuple[int, ...], tuple[slice, ...]]]: ...


def get_chunked_slices(
    screen: tuple[int, ...], chunk_shape: tuple[int, ...], camera: tuple[int, ...]
) -> Iterator[tuple[tuple[slice, ...], tuple[int, ...], tuple[slice, ...]]]:
    """Iterate over map chunks covered by the screen.

    Args:
        screen: The shape of the :term:`screen`.
        chunk_shape: The shape of individual chunks.
        camera: The :term:`camera` position.

    Yields:
        ``(screen_slice, chunk_index, chunk_slice)``

        For the chunk at `chunk_index` it should be sliced with `chunk_slice` to match a screen sliced with
        `screen_slice`.

    Example::

        CHUNK_SIZE: tuple[int, int]
        screen: NDarray  # Screen array.
        chunks: dict[tuple[int, int], NDarray]  # Mapping of chunked arrays.
        camera: tuple[int, int]
        for screen_slice, chunk_ij, chunk_slice in tcod.camera.get_chunked_slices(screen.shape, CHUNK_SIZE, camera):
            if chunk_ij in chunks:
                screen[screen_slice] = chunks[chunk_ij][chunk_slice]

    >>> list(get_chunked_slices((10,10),(10,10),(0,0)))
    [((slice(0, 10, None), slice(0, 10, None)), (0, 0), (slice(0, 10, None), slice(0, 10, None)))]
    >>> list(get_chunked_slices((10,10),(10,10),(-5,-5)))
    [((slice(0, 5, None), slice(0, 5, None)), (-1, -1), (slice(5, 10, None), slice(5, 10, None))), ((slice(0, 5, None), slice(5, 10, None)), (-1, 0), (slice(5, 10, None), slice(0, 5, None))), ((slice(5, 10, None), slice(0, 5, None)), (0, -1), (slice(0, 5, None), slice(5, 10, None))), ((slice(5, 10, None), slice(5, 10, None)), (0, 0), (slice(0, 5, None), slice(0, 5, None)))]
    """
    assert len(screen) == len(chunk_shape) == len(camera)
    chunk_lines = (
        _get_chunked_slices_1d(screen_, chunk_, camera_)
        for screen_, chunk_, camera_ in zip(screen, chunk_shape, camera)
    )
    for c in itertools.product(*chunk_lines):
        out_screen, out_id, out_chunk = zip(*c)
        yield out_screen, out_id, out_chunk


@overload
def get_camera(
    screen: tuple[int],
    center: tuple[int],
    clamping: tuple[tuple[int], float | tuple[float]] | None = None,
) -> tuple[int]: ...
@overload
def get_camera(
    screen: tuple[int, int],
    center: tuple[int, int],
    clamping: tuple[tuple[int, int], float | tuple[float, float]] | None = None,
) -> tuple[int, int]: ...
@overload
def get_camera(
    screen: tuple[int, int, int],
    center: tuple[int, int, int],
    clamping: tuple[tuple[int, int, int], float | tuple[float, float, float]] | None = None,
) -> tuple[int, int, int]: ...
@overload
def get_camera(
    screen: tuple[int, ...],
    center: tuple[int, ...],
    clamping: tuple[tuple[int, ...], float | tuple[float, ...]] | None = None,
) -> tuple[int, ...]: ...


def get_camera(
    screen: tuple[int, ...],
    center: tuple[int, ...],
    clamping: tuple[tuple[int, ...], float | tuple[float, ...]] | None = None,
) -> tuple[int, ...]:
    """Return the translation position for the camera from the given center position, screen size, and clamping rule.

    Args:
        screen: The :term:`screen` shape.
        center: The :term:`world` position which the camera will center on.
        clamping: The clamping rules, this is ``(world, justify)`` as if provided to :any:`clamp_camera`.
            If clamping is `None` then this function only does the minimum of subtracting half the screen size to get
            the camera position.

            `world` is the world shape.
            `justify` can be `(0.5, 0.5)` to center the world when it's smaller than the camera,
            or `(0, 0)` to place the world towards zero.  This would be the upper-left corner with libtcod.

    Returns:
        The clamped :term:`camera` position.
    """
    assert len(screen) == len(center)
    camera = tuple(center_ - screen_ // 2 for center_, screen_ in zip(center, screen))
    if clamping is not None:
        world, justify = clamping
        camera = clamp_camera(screen, world, camera, justify)
    return camera
