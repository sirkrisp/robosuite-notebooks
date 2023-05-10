from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class Pyramid3DEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        floor_1 = BrickObj(
            name="floor_1",
            num_segments_x=7,
            num_segments_y=7,
        )
        floor_2 = BrickObj(
            name="floor_2",
            num_segments_x=5,
            num_segments_y=5,
        )
        floor_3 = BrickObj(
            name="floor_3",
            num_segments_x=3,
            num_segments_y=3,
        )
        floor_4 = BrickObj(
            name="floor_4",
            num_segments_x=1,
            num_segments_y=1,
        )

        return [floor_1, floor_2, floor_3, floor_4]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (1, 1), 1, (0, 0), 0),
            BrickAssemblyInstruction(1, (1, 1), 2, (0, 0), 0),
            BrickAssemblyInstruction(2, (1, 1), 3, (0, 0), 0),
        ]
