import sys
import os
import tqdm
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append('../../metaworld')
# sys.path.append('..')

# from bricks_dataset.env_dict import ALL_BRICK_ENVIRONMENTS
from bricks_dataset.brick_envs.generative_env import GenerativeEnv
from bricks_dataset.brick_objects.brick_colors import hsl_colors
from datagen.datagen_utils import NpEncoder
import copy


def try_until_success(func, max_tries=100):
    for i in range(max_tries):
        try:
            return func()
        except:
            # TODO check if keyboard interrupt
            continue


# TODO improve performance
# copied from test_datagen.ipynb
def generate_brick_configs(num_brick_shapes, num_blocks, memory):
    """ Generates all possible brick configurations for a given number of brick shapes and blocks. brick_config[i] is the number of bricks of shape i (index starts at 1).
    """
    if memory[num_brick_shapes, num_blocks] is not None:
        return memory[num_brick_shapes, num_blocks]
    if num_blocks == 0:
        shape = {}
        for i in range(1, num_brick_shapes+1):
            shape[i] = 0
        memory[num_brick_shapes, num_blocks] = [shape]
    elif num_brick_shapes == 1:
        # only one option left
        memory[num_brick_shapes, num_blocks] = [{1: num_blocks}]
    else:
        all_block_shapes = []
        for i in range(num_blocks + 1):
            # NOTE we need deepcopy here because otherwise we will modify the same dictionary
            block_shapes = copy.deepcopy(generate_brick_configs(num_brick_shapes - 1, num_blocks - i, memory))
            # TODO not so optimal => makes the algorithm super slow
            for other_block_shape in block_shapes:
                other_block_shape[num_brick_shapes] = i
            all_block_shapes += block_shapes
        memory[num_brick_shapes, num_blocks] = all_block_shapes

    return memory[num_brick_shapes, num_blocks]



if __name__ == "__main__":

    output_folder = os.path.join(os.path.dirname(__file__), "manuals_10")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ## Params

    # brick params
    num_blocks = 7
    block_sizes = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (2, 2, 1), (2, 3, 1), (2, 4, 1), (3, 3, 1), (3, 4, 1), (4, 4, 1)]

    # generative env params
    show_segments = True
    
    # sample params
    num_envs = 100
    num_samples_per_env = 100 # corresponds to num samples per task env
    sample_colors = False

    # render params
    camera_name = "frontview"
    img_res = [300, 300]

    # create brick shape configurations
    num_block_sizes = len(block_sizes)
    memory = np.empty((num_block_sizes + 1, num_blocks + 1), dtype="object")
    brick_configs = generate_brick_configs(num_block_sizes, num_blocks, memory)
    print(f"There are {len(brick_configs)} brick configurations with {num_blocks} blocks and {num_block_sizes} block sizes.")
    # sample num_envs brick configurations
    brick_configs = np.random.choice(brick_configs, size=num_envs, replace=False)
    # convert to list of brick shape dicts
    all_brick_shapes = []
    for i in range(len(brick_configs)):
        brick_shapes = {}
        brick_index = 0
        brick_config = brick_configs[i]
        for k in brick_config.keys():
            # NOTE keys of brick_config start at 1
            block_size = block_sizes[int(k) - 1]
            num_blocks_of_block_size = brick_config[k]
            for b in range(num_blocks_of_block_size):
                brick_shapes[brick_index] = block_size
                brick_index += 1
        all_brick_shapes.append(brick_shapes)

    # image array
    images = np.memmap(
        filename=os.path.join(output_folder, "keyframe_images.data"),
        dtype=np.uint8,
        mode="w+",
        shape=(num_envs, num_samples_per_env, num_blocks, *img_res, 3),
    )

    # text instructions
    manuals = []

    # info
    info = {}
    info["data"] = {
        "images": {
            "shape": images.shape,
        },
        "manuals": manuals
    }
    info["meta"] = {
        # brick params
        "num_blocks": num_blocks,
        "block_sizes": block_sizes,

        # generative env params
        "all_brick_shapes": all_brick_shapes,
        "show_segments": show_segments,

        # sample params
        "num_envs": num_envs,
        "num_samples_per_env": num_samples_per_env,
        "sample_colors": sample_colors,

        # render params
        "camera_name": camera_name,
        "img_res": img_res,
    }

    keyframes_start_index = 0
    for i in tqdm.tqdm(range(num_envs)):
        env_manuals = []
        brick_shapes = all_brick_shapes[i]
        env = try_until_success(lambda: GenerativeEnv(brick_shapes))
        env.hard_reset = False  # if hard_reset=True, env.reset() reinitializes MjModel from xml string which is very slow
        # sample colors
        brick_colors = ["custom_blue_grey" for _ in range(num_blocks + 1)] if not sample_colors else env.get_random_brick_colors()
        env.set_brick_colors(brick_colors)
        for j in range(num_samples_per_env):
            env.setup_instructions()
            try_until_success(lambda: env.reset())
            keyframes_data = env.get_keyframe_data(img_resolution=img_res, camera_name=camera_name)
            images[i, j] = np.array(keyframes_data["images"], dtype=np.uint8)
            # manual
            ordered_blocks = []
            instructions = []
            block_idx = env.instructions[0].brick_1_idx
            block = env.bricks[block_idx]
            block_idx_map = {block_idx: 0}
            ordered_blocks.append(
                (block.num_segments_x, block.num_segments_y)
            )
            for k in range(len(env.instructions)):
                env_instruction = env.instructions[k]
                block_idx = env_instruction.brick_2_idx
                block = env.bricks[block_idx]
                block_idx_map[block_idx] = k + 1
                ordered_blocks.append(
                    (block.num_segments_x, block.num_segments_y)
                )
                instructions.append((
                    block_idx_map[env_instruction.brick_1_idx],
                    block_idx_map[block_idx],  # unnecessary because blocks are ordered
                    # save offset between left bottom corner of block 1 and left bottom corner of block 2
                    tuple(np.array(env_instruction.pin_2) - np.array(env_instruction.pin_1))
                ))
            env_manuals.append({
                "ordered_blocks": ordered_blocks,
                "instructions": instructions,
            })
        manuals.append(env_manuals)
        env.close()

    images.flush()
    json.dump(info, open(os.path.join(output_folder, "info.json"), "w"), cls=NpEncoder)
