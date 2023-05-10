

from collections.abc import Iterable

import numpy as np

from robosuite.models.objects import CompositeObject, BoxObject
from robosuite.utils.mjcf_utils import BLUE, CYAN, GREEN, RED, CustomMaterial, add_to_dict


class Wall(CompositeObject):
    """
    A wall object.
    """

    def __init__(
            self,
            color=(1, 0, 0),
            size=(0.1, 0.1, 0.1),
            num_segments_width=5,
            num_segments_height=3,
            name="wall",
    ):
        super().__init__(name="wall")

        self.color = color
        self.size = size
        self.num_segments_width = num_segments_width
        self.num_segments_height = num_segments_height

        self._create_wall()

    def _create_wall(self):
        """
        Creates the wall object.
        """
        # Add wall segments
        for i in range(self.num_segments_width):
            for j in range(self.num_segments_height):
                self.add_wall_segment(i, j)

    def add_wall_segment(self, i, j):
        """
        Adds a wall segment to the wall.

        Args:
            i (int): The x index of the wall segment.
            j (int): The y index of the wall segment.
        """
        # Create wall segment
        wall_segment = BoxObject(
            name="wall_segment_{}_{}".format(i, j),
            size=self.size,
            rgba=self.color,
            density=1000,
            material=CustomMaterial(
                texture="",
                tex_name="",
                mat_name="",
                reflectance=0.0,
                rgba=self.color,
                tex_friction=1.0,
                tex_type="cube",
                tex_density=1000,
                tex_solref=None,
                tex_solimp=None,
            ),
        )

        # Add wall segment to wall
        self.merge_assets(wall_segment)
        self.worldbody.append(wall_segment.get_obj())

        # Add wall segment to wall segment dict
        add_to_dict(self.worldbody, "body", wall_segment.get_obj())

        # Add wall segment to wal

