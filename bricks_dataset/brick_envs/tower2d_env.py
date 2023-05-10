from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class Tower2DEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_x=1,
            num_segments_y=3,
            segment_height=0.01,
        )
        pillar_1_1 = BrickObj(
            name="pillar_1_1",
            num_segments_z=2,
        )
        pillar_1_2 = BrickObj(
            name="pillar_1_2",
            num_segments_z=2,
        )
        bar_1 = BrickObj(
            name="bar_1",
            num_segments_x=1,
            num_segments_y=3
        )
        pillar_2_1 = BrickObj(
            name="pillar_2_1",
            num_segments_z=2,
        )
        pillar_2_2 = BrickObj(
            name="pillar_2_2",
            num_segments_z=2,
        )
        bar_2 = BrickObj(
            name="bar_2",
            num_segments_x=1,
            num_segments_y=3
        )
        antenna = BrickObj(
            name="antenna",
        )

        return [base, pillar_1_1, pillar_1_2, bar_1, pillar_2_1, pillar_2_2, bar_2, antenna]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, 0, 1, 0, 0),
            BrickAssemblyInstruction(0, 2, 2, 0, 0),
            BrickAssemblyInstruction(1, 0, 3, 0, 0),

            BrickAssemblyInstruction(3, 0, 4, 0, 0),
            BrickAssemblyInstruction(3, 2, 5, 0, 0),
            BrickAssemblyInstruction(4, 0, 6, 0, 0),

            BrickAssemblyInstruction(6, 1, 7, 0, 0),
        ]
