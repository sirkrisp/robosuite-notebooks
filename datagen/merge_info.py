import os
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datagen.datagen_utils import NpEncoder


def merge_info(data_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # TODO also merge props other than manuals
    merged_manuals = []
    merged_info = {}
    merged_info["data"] = {
        "manuals": merged_manuals
    }

    for data_folder in data_folders:
        info_path = f"{data_folder}/info.json"
        # see generate_img_instructions_data.py for info on the info.json file
        info = json.load(open(info_path, "r"))
        manuals = info["data"]["manuals"]
        merged_manuals.extend(manuals)

    json.dump(merged_info, open(os.path.join(output_folder, "info.json"), "w"), cls=NpEncoder)


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
    output_folder = "/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_01-10"
    merge_info(
        data_folders=data_folders, 
        output_folder=output_folder,
    )