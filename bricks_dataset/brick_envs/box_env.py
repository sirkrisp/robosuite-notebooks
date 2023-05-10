from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class BoxEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_x=5,
            num_segments_y=5,
        )
        side_1 = BrickObj(
            name="side_1",
            num_segments_y=3,
            num_segments_z=2,
        )
        side_2 = BrickObj(
            name="side_2",
            num_segments_y=3,
            num_segments_z=2,
        )
        side_3 = BrickObj(
            name="side_3",
            num_segments_x=5,
            num_segments_z=2,
        )
        side_4 = BrickObj(
            name="side_4",
            num_segments_x=5,
            num_segments_z=2,
        )

        return [base, side_1, side_2, side_3, side_4]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (0, 1), 1, 0, 0),
            BrickAssemblyInstruction(0, (4, 1), 2, 0, 0),
            BrickAssemblyInstruction(0, (0, 0), 3, 0, 0),
            BrickAssemblyInstruction(0, (0, 4), 4, 0, 0),
        ]
