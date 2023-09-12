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


def merge_dataset_info_1_into_info_2(info_1, info_2, prefix=""):
    total_num_images_1 = info_1["dataset"]["total_num_images"]
    total_num_images_2 = info_2["dataset"]["total_num_images"]
    info_1["dataset"]["total_num_images"] += total_num_images_2
    for task in info_2["tasks"].keys():
        new_task = info_2["tasks"][task]
        new_task_name = prefix + task
        for sample in new_task["samples"]:
            sample["keyframes_start_index"] += total_num_images_1
        info_1["tasks"][new_task_name] = new_task


if __name__ == "__main__":

    output_folder = os.path.join(os.path.dirname(__file__), "clip_data_05_06")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset_folders = [
        os.path.join(os.path.dirname(__file__), "clip_data_05_pairs_seg_numBricks=7"),
        os.path.join(os.path.dirname(__file__), "clip_data_06_pairs_seg_numBricks=6"),
        # os.path.join(os.path.dirname(__file__), "keyframes_8"),
        # os.path.join(os.path.dirname(__file__), "keyframes_9"),
    ]

    # TODO merged info should have array of configs
    infos = []
    images = []
    total_num_images = 0
    for dataset_folder in dataset_folders:
        info = json.load(open(os.path.join(dataset_folder, "info.json"), "r"))
        infos.append(info)
        total_num_images += info["dataset"]["total_num_images"]
        img_res = info["dataset"]["img_res"]
        assert img_res[0] == 300 and img_res[1] == 300
        image_arr = np.memmap(
            os.path.join(dataset_folder, "keyframe_images.data"),
            dtype=np.uint8,
            mode='r',
            shape=(2, info["dataset"]["total_num_images"], 3, *img_res)
        )
        images.append(image_arr)

    merged_info = infos[0]
    merged_images = np.memmap(
        filename=os.path.join(output_folder, "keyframe_images.data"),
        dtype=np.uint8,
        mode="w+",
        shape=(2, total_num_images, 3, 300, 300),
    )
    for i in range(1, len(infos)):
        merge_dataset_info_1_into_info_2(merged_info, infos[i], prefix=f"{i}_")

    for p in range(2):
        merged_images[p][0:images[0][p].shape[0]] = images[0][p]
        counter = images[0][p].shape[0]
        for i in range(1, len(infos)):
            merged_images[p][counter:counter+images[i][p].shape[0]] = images[i][p]
            counter += images[i][p].shape[0]

    merged_images.flush()
    # depths.flush()
    json.dump(merged_info, open(os.path.join(output_folder, "info.json"), "w"), cls=NpEncoder)
