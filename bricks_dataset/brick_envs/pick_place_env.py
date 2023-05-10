from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


class PickPlaceEnv(BricksBaseEnv):

    def __init__(self):
        super().__init__()

    def _create_bricks(self):
        dest_1 = BrickObj(
            name="dest_1",
            num_segments_x=3,
            num_segments_y=3,
        )
        dest_2 = BrickObj(
            name="dest_2",
            num_segments_x=3,
            num_segments_y=3,
        )
        dest_3 = BrickObj(
            name="dest_3",
            num_segments_x=3,
            num_segments_y=3,
        )
        obj_1 = BrickObj(
            name="obj_1"
        )
        obj_2 = BrickObj(
            name="obj_2"
        )
        obj_3 = BrickObj(
            name="obj_3"
        )

        return [dest_1, dest_2, dest_3, obj_1, obj_2, obj_3]

    def _create_instructions(self):
        return [
            BrickAssemblyInstruction(0, (1, 1), 3, 0, 0),
            BrickAssemblyInstruction(1, (1, 1), 4, 0, 0),
            BrickAssemblyInstruction(2, (1, 1), 5, 0, 0),
        ]
