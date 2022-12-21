Glossary
========

.. glossary::

    Camera
        This is a vector used to translate between :term:`screen`-space and :term:`world`-space coordinates.

        The camera is positioned in world-space.
        :term:`Screen` position ``(0, 0)`` is projected onto the :term:`world` position at where the camera is placed.
        This is the upper-left corner of the screen in libtcod or SDL.

        Once you get the camera position via :any:`get_camera` or by manually placing it you can convert between screen coordinates and world coordinates by applying vector math.
        This position is also used by :any:`get_slices`, :any:`get_views`, or :any:`get_chunked_slices`.

        Add the camera position to a screen position (such as a mouse tile position) to get the world position (such as where in the world itself the mouse is hovering over.)
        Subtract the camera position from a world position (such as a player object position) to get the screen position (such as where to draw the player on the screen.)

        See the `RogueBasin article on Scrolling maps <http://roguebasin.com/index.php/Scrolling_map>`_ for more details and a visual example.

    Screen
        Screen-space is the array which is projected into the :term:`world` using a :term:`camera`.

        Normally this array is something like a tcod console, such as :any:`tcod.console.Console.rgb`.
        However, this can be any temporary array projected into the world.

    World
        This is the map data which is stored normally as one or more arrays.
