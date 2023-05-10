# from robosuite.models.objects import BoxObject

from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class TestEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        cube_1 = BrickObj(
            name="cube_1",
            num_segments_y=4,
            # hsl=(1, 0, 0, 1)
            # material=redwood
        )
        cube_2 = BrickObj(
            name="cube_2",
            num_segments_y=3,
            # hsl=(0, 1, 0, 1),
            # segment_height=0.01
        )
        cube_3 = BrickObj(
            name="cube_3",
            num_segments_y=4,
            num_segments_x=4,
            # hsl=(0, 0, 1, 1),
            segment_height=0.01
        )
        cube_4 = BrickObj(
            name="cube_4",
            num_segments_y=5,
            num_segments_x=2,
            # hsl=(0, 0, 1, 1),
        )
        cube_5 = BrickObj(
            name="cube_5",
            num_segments_y=4,
            num_segments_x=1,
            num_segments_z=3,
            # hsl=(0, 1, 1, 1),
        )
        return [cube_1, cube_2, cube_3, cube_4, cube_5]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(2, (3, 3), 0, 0, 3),
            BrickAssemblyInstruction(2, (1, 0), 1, 0, 0)
        ]
