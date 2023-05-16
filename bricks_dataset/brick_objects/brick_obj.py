from __future__ import annotations

from robosuite.models.objects import CompositeObject
import numpy as np

from bricks_dataset.brick_objects.brick_colors import hsl_to_rgba, hsl_change_brightness


class BrickObj(CompositeObject):

    def __init__(
            self,
            name: str,
            hsl=(204, 8, 76),
            num_segments_y=1,
            num_segments_x=1,
            num_segments_z=1,
            segment_height=0.025,
            segment_size=0.025,
            show_orientation=True,
            **kwargs
    ):
        self.num_segments_y = num_segments_y
        self.num_segments_x = num_segments_x
        self.num_segments_z = num_segments_z
        self.show_orientation = show_orientation

        # self.segment_height_half = segment_height / 2
        self.segment_size_half = segment_size / 2

        self.size_x_half = num_segments_x * segment_size / 2
        self.size_y_half = num_segments_y * segment_size / 2
        self.size_z_half = num_segments_z * segment_height / 2

        rgba, rgba_bright, rgba_dark = self._get_colors(hsl)

        orient_depth_half = segment_size / 2 / 2

        total_num_pins = num_segments_x * num_segments_y
        pin_half_sizes = [[segment_size / 2, segment_size / 2, segment_height / 2 * 0.1] for _ in range(total_num_pins)]
        pin_locations = []
        pin_z_half = segment_height / 2 * 0.1
        for i in range(num_segments_x):
            for j in range(num_segments_y):
                pin_locations.append(
                    [
                        -self.size_x_half + segment_size / 2 + i * segment_size,
                        -self.size_y_half + segment_size / 2 + j * segment_size,
                        self.size_z_half - pin_z_half
                    ]
                )
        pin_colors = self._get_pin_colors(hsl)

        # orientation_marker
        orientation_marker_half_size = [segment_size / 2, segment_size / 4, segment_height / 2 * 0.1]
        orientation_marker_location = pin_locations[0][:]  # copy
        orientation_marker_location[1] -= segment_size / 4
        orientation_marker_rgba = [1, 0, 0, 1]

        # adjust first pin for orientation marker
        pin_half_sizes[0][1] = orientation_marker_half_size[1]  # adjust y for orientation marker
        pin_locations[0][1] += segment_size / 4  # adjust y for orientation marker

        super().__init__(
            name=name,
            density=200,
            total_size=[self.size_x_half, self.size_y_half, self.size_z_half],
            locations_relative_to_center=True,
            geom_types=["box", "box", "box", *(["box"] * total_num_pins), "box"],
            geom_sizes=[
                [self.size_x_half - orient_depth_half, self.size_y_half - orient_depth_half, self.size_z_half - pin_z_half],
                [self.size_x_half, orient_depth_half, self.size_z_half - pin_z_half],
                [orient_depth_half, self.size_y_half - orient_depth_half, self.size_z_half - pin_z_half],
                *pin_half_sizes,
                orientation_marker_half_size
            ],
            geom_rgbas=[rgba, rgba, rgba, *pin_colors, orientation_marker_rgba],
            geom_locations=[
                (orient_depth_half, orient_depth_half, -pin_z_half),
                (0, -self.size_y_half + orient_depth_half, -pin_z_half),
                (-self.size_x_half + orient_depth_half, orient_depth_half, -pin_z_half),
                *pin_locations,
                orientation_marker_location,
            ],
            **kwargs
        )

    def _get_colors(self, hsl):
        l_dark = 0.75 if self.show_orientation else 1
        l_bright = 1.25 if self.show_orientation else 1
        hsl_dark = hsl_change_brightness(hsl, l_dark)
        hsl_bright = hsl_change_brightness(hsl, l_bright)
        return list(map(hsl_to_rgba, [hsl, hsl_bright, hsl_dark]))

    def _get_pin_colors(self, hsl):
        colors = self._get_colors(hsl)
        pin_colors = [colors[1] if (i + j) % 2 == 0 else colors[2] for i in range(self.num_segments_y) for j in
                      range(self.num_segments_x)]
        # pin_colors[0] = self._get_first_pin_color(hsl)
        return pin_colors

    def set_hsl(self, hsl: tuple[int, int, int]):
        colors = self._get_colors(hsl)
        pin_colors = self._get_pin_colors(hsl)
        box_colors = [colors[0], colors[0], colors[0], *pin_colors]  # *colors
        for i in range(len(box_colors)):
            self.geom_rgbas[i] = box_colors[i]
        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()
        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def pin_idx_to_pin_tuple(self, pin_idx: int | tuple[int, int]):
        if isinstance(pin_idx, tuple):
            pin_idx_x, pin_idx_y = pin_idx
        else:
            if self.num_segments_x == 1:
                pin_idx_y = pin_idx
                pin_idx_x = 0
            elif self.num_segments_y == 1:
                pin_idx_y = 0
                pin_idx_x = pin_idx
            else:
                raise ValueError("Invalid pin index")
        assert 0 <= pin_idx_y < self.num_segments_y, "Invalid pin index y"
        assert 0 <= pin_idx_x < self.num_segments_x, "Invalid pin index x"
        return pin_idx_x, pin_idx_y

    def get_pin_pos_local(self, pin_type: str, pin_idx: int | tuple[int, int]):
        pin_idx_x, pin_idx_y = self.pin_idx_to_pin_tuple(pin_idx)
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
