if __name__ == "__main__":
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("parent_dir", parent_dir)
    sys.path.append(str(parent_dir))

import os
import torch
from tqdm import tqdm
import json
import numpy as np

def merge_images(data_folders: str, output_folder: str):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    global_i = 0
    for data_folder in data_folders:
        print("data_folder", data_folder)
        # setup paths
        images_filename = "keyframe_images.data"
        info_filename = "info.json"
        images_path = os.path.join(data_folder, images_filename)
        info_path = os.path.join(data_folder, info_filename)

        # load info
        info = json.load(open(info_path, "r"))
        images_array_shape = info["data"]["images"]["shape"]
        assert len(images_array_shape) == 6  # (n_envs, n_samples_per_env, n_keyframes, img_res_x, img_res_y, n_channels)

        # load images
        images = np.array(np.memmap(
            images_path,
            dtype=np.uint8,
            mode='r',
            shape=tuple(images_array_shape)
        )[:])

        for i in tqdm(range(images.shape[0])):
            for j in range(images.shape[1]):
                keyframes = images[i][j]
                image_tokens_path = os.path.join(output_folder, f"keyframes_{global_i}.tar")
                torch.save(torch.from_numpy(keyframes), image_tokens_path)
                global_i += 1

if __name__ == "__main__":
    # merge_manuals(no_sep=False)
    data_folders = [
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_01",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_02",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_03",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_04",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_05",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_06",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_07",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_08",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_09",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_10",
    ]
    output_folder = "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_01-10/keyframes"
    merge_images(
        data_folders=data_folders, 
        output_folder=output_folder,
    )