from __future__ import annotations

from robosuite.models.objects import BoxObject, CompositeObject
import numpy as np


class BrickObj(CompositeObject):

    def __init__(
            self,
            name: str,
            color=(1, 0, 0, 1),
            num_segments_y=1,
            num_segments_x=1,
            num_segments_z=1,
            segment_height=0.02,
            segment_size=0.02,
            show_orientation=True,
            **kwargs
    ):
        self.num_segments_y = num_segments_y
        self.num_segments_x = num_segments_x
        self.num_segments_z = num_segments_z

        # self.segment_height_half = segment_height / 2
        self.segment_size_half = segment_size / 2

        self.size_x_half = num_segments_x * segment_size / 2
        self.size_y_half = num_segments_y * segment_size / 2
        self.size_z_half = num_segments_z * segment_height / 2

        orient_depth_half = segment_size / 2 / 2

        color_dark_factor = 0.5 if show_orientation else 1
        color_brightness_factor = 4 if show_orientation else 1
        color_dark = (color_dark_factor * np.array(color[:3])).tolist() + [1]
        color_bright = (color_brightness_factor * np.array(color[:3])).tolist() + [1]

        super().__init__(
            name=name,
            density=200,
            total_size=[self.size_x_half, self.size_y_half, self.size_z_half],
            locations_relative_to_center=True,
            geom_types=["box", "box", "box"],
            geom_sizes=[
                [self.size_x_half - orient_depth_half, self.size_y_half - orient_depth_half, self.size_z_half],
                [self.size_x_half, orient_depth_half, self.size_z_half],
                [orient_depth_half, self.size_y_half - orient_depth_half, self.size_z_half]
            ],
            geom_rgbas=[color, color_bright, color_dark],
            geom_locations=[
                (orient_depth_half, orient_depth_half, 0),
                (0, -self.size_y_half + orient_depth_half, 0),
                (-self.size_x_half + orient_depth_half, orient_depth_half, 0)
            ],
            **kwargs
        )

    def get_pin_pos_local(self, pin_type: str, pin_idx: int | tuple[int, int]):
        if isinstance(pin_idx, tuple):
            pin_idx_x, pin_idx_y = pin_idx
        else:
            pin_idx_y = pin_idx
            pin_idx_x = 0
        assert 0 <= pin_idx_y < self.num_segments_y, "Invalid pin index y"
        assert 0 <= pin_idx_x < self.num_segments_x, "Invalid pin index x"

        pin_z_offset = self.size_z_half
        if pin_type == "bottom":
            pin_z_offset *= -1
        elif pin_type != "top":
            raise ValueError("Invalid pin type")

        return np.array([
            -self.size_x_half + (0.5 + pin_idx_x) * 2 * self.segment_size_half,
            -self.size_y_half + (0.5 + pin_idx_y) * 2 * self.segment_size_half,
            pin_z_offset
        ])

#
# class BrickObj(BoxObject):
#
#     def __init__(self, name: str, color=(1, 0, 0, 1), num_segments=1, segment_height=0.04, segment_size=0.04, **kwargs):
#         self.num_segments = num_segments
#         self.segment_height = segment_height
#         self.segment_size = segment_size
#
#         self.length = num_segments * segment_size
#
#         super().__init__(
#             name=name,
#             density=200,
#             size=[segment_size / 2, self.length / 2, segment_height / 2],
#             rgba=color,
#             **kwargs
#         )
#
#     def get_pin_pos_local(self, pin_idx: int, pin_type: str):
#         assert 0 <= pin_idx < self.num_segments, "Invalid pin index"
#
#         pin_z_offset = self.segment_height / 2
#         if pin_type == "bottom":
#             pin_z_offset *= -1
#         elif pin_type != "top":
#             raise ValueError("Invalid pin type")
#
#         return np.array([0, -self.length / 2 + (0.5 + pin_idx) * self.segment_size, pin_z_offset])
