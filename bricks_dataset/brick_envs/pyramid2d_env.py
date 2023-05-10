from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class Pyramid2DEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_y=6,
            segment_height=0.01,
        )
        brick_1_1 = BrickObj(
            name="brick_1_1",
            num_segments_y=2,
        )
        brick_1_2 = BrickObj(
            name="brick_1_2",
            num_segments_y=2,
        )
        brick_1_3 = BrickObj(
            name="brick_1_3",
            num_segments_y=2,
        )
        brick_2_1 = BrickObj(
            name="brick_2_1",
            num_segments_y=2,
        )
        brick_2_2 = BrickObj(
            name="brick_2_2",
            num_segments_y=2,
        )
        brick_3_1 = BrickObj(
            name="brick_3_1",
            num_segments_y=2,
        )

        return [base, brick_1_1, brick_1_2, brick_1_3, brick_2_1, brick_2_2, brick_3_1]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, 0, 1, 0, 0),
            BrickAssemblyInstruction(0, 2, 2, 0, 0),
            BrickAssemblyInstruction(0, 4, 3, 0, 0),

            BrickAssemblyInstruction(1, 1, 4, 0, 0),
            BrickAssemblyInstruction(2, 1, 5, 0, 0),

            BrickAssemblyInstruction(4, 1, 6, 0, 0),
        ]
