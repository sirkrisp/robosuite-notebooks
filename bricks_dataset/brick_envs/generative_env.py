from .bricks_base_env import BricksBaseEnv
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction
from bricks_dataset.brick_envs.generative_utils import generate_instructions


class GenerativeEnv(BricksBaseEnv):

    def __init__(self, brick_shapes: dict[int, tuple[int, int, int]], show_segments: bool = True):
        """
        Args:
            brick_shapes: brick_shapes[i] is the shape of the ith brick
            show_segments: whether to show the segments of the bricks
        """
        self.brick_shapes = brick_shapes
        self.show_segments = show_segments
        super().__init__()

    def _create_bricks(self):
        bricks = []
        for k in list(self.brick_shapes.keys()):
            shape = self.brick_shapes[k]
            bricks.append(BrickObj(
                name=f"brick_{k}",
                num_segments_x=shape[0],
                num_segments_y=shape[1],
                num_segments_z=shape[2],
                show_segments=self.show_segments
            ))
        return bricks

    def _create_instructions(self):
        return generate_instructions(self.brick_shapes)

    # TODO does not work, need to call env.reset() to reinitialize MjModel which is very expensive
    # def resample(self):
    #     self.brick_shapes = generate_brick_shapes(self.num_blocks, self.total_num_pins, 1)
    #     self.brick_shapes[0] = (*self.base_shape, 1)
    #     self.setup_bricks()
    #     self._load_model(bricks_only=True)
    #     self.model.make_mappings()
    #     self._setup_references()
    #     self._reset_internal()
    #     self.setup_instructions()

