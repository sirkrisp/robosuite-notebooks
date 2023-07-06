from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction
from bricks_dataset.brick_envs.generative_utils import generate_brick_shapes, generate_instructions


class GenerativeEnv(BricksBaseEnv):

    def __init__(self, num_blocks: int, total_num_pins: tuple[int, int, int], base_shape: tuple[int, int], show_segments=True):
        """
        Args:
            num_blocks: number of blocks in the environment (does not include the base brick)
            total_num_pins: total number of pins in the environment (does not include the base brick)
            base_shape: shape of the base brick
            show_segments: whether to show the segments of the bricks
        """
        self.num_blocks = num_blocks
        self.total_num_pins = total_num_pins
        self.base_shape = base_shape
        self.show_segments = show_segments
        self.brick_shapes = generate_brick_shapes(self.num_blocks, self.total_num_pins, 1)
        self.brick_shapes[0] = (*self.base_shape, 1)
        super().__init__()

    def _create_bricks(self):
        bricks = []
        for i in range(len(self.brick_shapes.keys())):
            shape = self.brick_shapes[i]
            bricks.append(BrickObj(
                name=f"brick_{i}",
                num_segments_x=shape[0],
                num_segments_y=shape[1],
                num_segments_z=shape[2],
                show_segments=self.show_segments
            ))
        return bricks

    def _create_instructions(self):
        return generate_instructions(self.brick_shapes)
