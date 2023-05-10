import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append('../../metaworld')
# sys.path.append('..')

import tqdm
import numpy as np


from bricks_dataset.env_dict import ALL_BRICK_ENVIRONMENTS


if __name__ == "__main__":

    output_folder = f'./trajectories-v6'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tasks = list(ALL_BRICK_ENVIRONMENTS.keys())
    print("tasks: ", tasks)

    # for each of the 10 tasks in the ALL_BRICK_ENVIRONMENTS dict (see above),
    # generate 10 instructions for 100 colors. In total, 10 * 10 * 100 = 10000 instructions
    # Furthermore, each instruction has multiple steps, so we have roughly 10000 * 10 = 100000 images

    num_colors_per_env = 100
    num_samples_per_color = 10

    envs = []
    total_num_instructions = 0
    for i in range(len(tasks)):
        env = ALL_BRICK_ENVIRONMENTS[tasks[i]]()
        envs.append(env)
        total_num_instructions += env.num_instructions

    # we add +1 for the initial image
    total_num_images = (total_num_instructions + len(tasks)) * num_colors_per_env * num_samples_per_color
    total_num_depth_images = total_num_images

    # in the end we need to be able to draw two samples where the color of the bricks is the same
    # draw task
    # draw color



    for i in range(len(tasks)):
        env = ALL_BRICK_ENVIRONMENTS[tasks[i]]()
        for c in range(100):
            # TODO sample color
            for s in range(10):
                # TODO sample instruction
                task_data = env.get_keyframe_data(img_resolution=(300, 300))
                # TODO save task_data
                pass




