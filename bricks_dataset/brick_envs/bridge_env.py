from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class BridgeEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_x=2,
            num_segments_y=14,
            segment_height=0.01,
        )
        pillar_1 = BrickObj(
            name="pillar_1",
            num_segments_x=2,
            num_segments_y=2,
            num_segments_z=3,
        )
        pillar_2 = BrickObj(
            name="pillar_2",
            num_segments_x=2,
            num_segments_y=2,
            num_segments_z=3,
        )
        pillar_3 = BrickObj(
            name="pillar_3",
            num_segments_x=2,
            num_segments_y=2,
            num_segments_z=3,
        )
        seg_1 = BrickObj(
            name="seg_1",
            num_segments_x=2,
            num_segments_y=6,
        )
        seg_2 = BrickObj(
            name="seg_2",
            num_segments_x=2,
            num_segments_y=6,
        )

        return [base, pillar_1, pillar_2, pillar_3, seg_1, seg_2]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (0, 0), 1, (0, 0), 0),
            BrickAssemblyInstruction(0, (0, 6), 2, (0, 0), 0),

            BrickAssemblyInstruction(1, (0, 1), 4, (0, 0), 0),

            BrickAssemblyInstruction(0, (0, 12), 3, (0, 0), 0),

            BrickAssemblyInstruction(2, (0, 1), 5, (0, 0), 0),
        ]
