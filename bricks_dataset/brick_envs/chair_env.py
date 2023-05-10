from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class ChairEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        base = BrickObj(
            name="base",
            num_segments_x=4,
            num_segments_y=6,
            segment_height=0.01,
        )
        leg_1 = BrickObj(
            name="leg_1",
            num_segments_x=4,
            num_segments_y=1,
            num_segments_z=3,
        )
        leg_2 = BrickObj(
            name="leg_2",
            num_segments_x=4,
            num_segments_y=1,
            num_segments_z=3,
        )
        seat_1 = BrickObj(
            name="seat_1",
            num_segments_x=2,
            num_segments_y=6,
        )
        seat_2 = BrickObj(
            name="seat_2",
            num_segments_x=2,
            num_segments_y=6,
        )
        back = BrickObj(
            name="back",
            num_segments_x=1,
            num_segments_y=6,
            num_segments_z=3,
        )

        return [base, leg_1, leg_2, seat_1, seat_2, back]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (0, 0), 1, 0, 0),
            BrickAssemblyInstruction(0, (0, 5), 2, 0, 0),
            BrickAssemblyInstruction(1, 0, 3, (0, 0), 0),
            BrickAssemblyInstruction(1, 2, 4, (0, 0), 0),
            BrickAssemblyInstruction(4, (1, 0), 5, 0, 0),
        ]
