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

if __name__ == "__main__":

    output_folder = os.path.join(os.path.dirname(__file__), "keyframes_10")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # tasks = list(ALL_BRICK_ENVIRONMENTS.keys())
    # print("tasks: ", tasks)

    # for each of the 10 tasks in the ALL_BRICK_ENVIRONMENTS dict (see above),
    # generate instructions for 100 colors. In total, 10 * 100 = 1000 instructions
    # Furthermore, each instruction has multiple steps, so we have roughly 10000 * 10 = 10000 images

    # num_colors_per_env = 100

    info = {}

    # envs = []

    # generative env params
    num_bricks = 6  # does not include base brick
    total_num_pins = (1, 24, 1)
    base_shape = (4, 4)

    # sample params
    total_num_env_samples = 1000
    num_colors_per_env = 1

    # total_num_instructions = 0
    # for i in range(len(tasks)):
    #     env = ALL_BRICK_ENVIRONMENTS[tasks[i]]()
    #     # envs.append(env)
    #     total_num_instructions += env.num_instructions
    #     # env.close()

    total_num_instructions = total_num_env_samples * num_colors_per_env * num_bricks
    # we add +1 for the initial image
    # total_num_images = (total_num_instructions + len(tasks)) * num_colors_per_env
    total_num_images = total_num_env_samples * num_colors_per_env * (num_bricks + 1)
    print("total_num_images: ", total_num_images)

    img_res = [300, 300]
    images = np.memmap(
        filename=os.path.join(output_folder, "keyframe_images.data"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_num_images, 3, *img_res),
    )

    # TODO add versioning
    info["tasks"] = {}
    info["dataset"] = {
        "max_pin": 8,
        "max_orientation": 4,
        "color_keys": list(hsl_colors.keys()),
        "img_res": img_res,
        "total_num_images": total_num_images
    }

    keyframes_start_index = 0
    for i in tqdm.tqdm(range(total_num_env_samples)):
        # TODO initialize generative env
        # env = ALL_BRICK_ENVIRONMENTS[tasks[i]]()
        env = None
        while True:
            try:
                env = GenerativeEnv(num_bricks, total_num_pins, base_shape)
                break
            except:
                continue
        num_instructions = env.num_instructions
        instructions = []
        for instruction in env.instructions:
            brick_1 = env.bricks[instruction.brick_1_idx]
            brick_2 = env.bricks[instruction.brick_2_idx]
            pin_1_x, pin_1_y = brick_1.pin_idx_to_pin_tuple(instruction.pin_1)
            pin_2_x, pin_2_y = brick_2.pin_idx_to_pin_tuple(instruction.pin_2)
            instructions.append({
                "brick_1": instruction.brick_1_idx,
                "brick_2": instruction.brick_2_idx,
                "pin_1_x": pin_1_x,
                "pin_1_y": pin_1_y,
                "pin_2_x": pin_2_x,
                "pin_2_y": pin_2_y,
                "orientation": instruction.o,
            })
        samples = []
        for c in range(num_colors_per_env):
            # sample color
            brick_colors = env.get_random_brick_colors()
            samples.append({
                "brick_colors": brick_colors,
                "keyframes_start_index": keyframes_start_index
            })
            env.set_brick_colors(brick_colors)
            while True:
                # sample task
                try:
                    env.reset()
                    break
                except:
                    pass
            # sample keyframe data
            keyframe_data = env.get_keyframe_data(img_resolution=(img_res[0], img_res[1]), camera_name="birdview")
            # save task_data
            images[keyframes_start_index:keyframes_start_index + num_instructions + 1] = \
                np.moveaxis(keyframe_data['images'], 3, 1).astype(np.uint8)
            # depths[img_start_index:img_start_index+num_instructions+1] = keyframe_data['depths']

            keyframes_start_index += num_instructions + 1

        task_name = f"task_{i}"
        info["tasks"][task_name] = {
            "samples": samples,
            "instructions": instructions,
            "brick_shapes": env.brick_shapes,  # {k: env.brick_shapes[k] for k in env.brick_shapes.keys()},
        }
        env.close()

    images.flush()
    # depths.flush()
    json.dump(info, open(os.path.join(output_folder, "info.json"), "w"), cls=NpEncoder)
