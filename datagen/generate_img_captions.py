import os
import json
from tqdm import tqdm

def generate_img_captions(data_folder, only_block_offset):

    # data_folder = os.path.join(os.path.dirname(__file__), "manuals_01")
    info_path = os.path.join(data_folder, "info.json")
    output_path = os.path.join(data_folder, "captions_full_no_sep.txt")

    # 1) load info from dataset
    info = json.load(open(info_path, "r"))
    # see generate_img_instructions_data.py for info on the info.json file
    manuals = info["data"]["manuals"]

    # 2) create captions   
    captions = []
    for i in tqdm(range(len(manuals))):
        for j in range(len(manuals[i])):
            manual = manuals[i][j]
            ordered_blocks = manual["ordered_blocks"]
            ordered_blocks_with_id = {}
            for k in range(len(ordered_blocks)):
                ordered_blocks_with_id[k] = ordered_blocks[k]
            # caption = f"A: {ordered_blocks_with_id}."
            # caption += f" B: {manual['instructions']}."
            
            # no seps
            caption = f"A"
            for k in ordered_blocks_with_id:
                caption += f"{ordered_blocks_with_id[k][0]}{ordered_blocks_with_id[k][1]}"
            caption += f"B"
            for instruction in manual['instructions']:
                caption += f"{instruction[1]}{instruction[2][0]}{instruction[2][1]}"
            caption += f"."
            captions.append(caption + "\n")  # writelines does not add newline

    file = open(output_path,'w')
    file.writelines(captions)
    file.close()

if __name__ == "__main__":
    data_folders = [
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_01",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_02",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_03",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_04",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_05",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_06",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_07",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_08",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_09",
        "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_10",
    ]
    for data_folder in data_folders:
        print("data_folder", data_folder)
        generate_img_captions(data_folder)