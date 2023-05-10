from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class Tower3DEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_x=3,
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
        pillar_1_3 = BrickObj(
            name="pillar_1_3",
            num_segments_z=2,
        )
        pillar_1_4 = BrickObj(
            name="pillar_1_4",
            num_segments_z=2,
        )
        floor_1 = BrickObj(
            name="floor_1",
            num_segments_x=3,
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
        pillar_2_3 = BrickObj(
            name="pillar_2_3",
            num_segments_z=2,
        )
        pillar_2_4 = BrickObj(
            name="pillar_2_4",
            num_segments_z=2,
        )
        floor_2 = BrickObj(
            name="floor_2",
            num_segments_x=3,
            num_segments_y=3
        )
        antenna = BrickObj(
            name="antenna",
        )

        return [base, pillar_1_1, pillar_1_2, pillar_1_3, pillar_1_4, floor_1,
                pillar_2_1, pillar_2_2, pillar_2_3, pillar_2_4, floor_2, antenna]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (0, 0), 1, 0, 0),
            BrickAssemblyInstruction(0, (0, 2), 2, 0, 0),
            BrickAssemblyInstruction(0, (2, 0), 3, 0, 0),
            BrickAssemblyInstruction(0, (2, 2), 4, 0, 0),
            BrickAssemblyInstruction(1, 0, 5, (0, 0), 0),

            BrickAssemblyInstruction(5, (0, 0), 6, 0, 0),
            BrickAssemblyInstruction(5, (0, 2), 7, 0, 0),
            BrickAssemblyInstruction(5, (2, 0), 8, 0, 0),
            BrickAssemblyInstruction(5, (2, 2), 9, 0, 0),
            BrickAssemblyInstruction(6, 0, 10, (0, 0), 0),

            BrickAssemblyInstruction(10, (1, 1), 11, 0, 0),
        ]
