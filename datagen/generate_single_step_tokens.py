import json
import torch
import os
from tqdm import tqdm

def generate_single_step_tokens(data_folders: list[str], output_folder: str):

    output_tokens_path = os.path.join(output_folder, "single_step_tokens.tar")

    chars = ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "*"]
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")
    stoi = { ch:i for i,ch in enumerate(chars) }
    max_caption_length = 4  # sample caption: -1-2

    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    tokens = []
    for data_folder in data_folders:
        info_path = f"{data_folder}/info.json"
        info = json.load(open(info_path, "r"))
        # see generate_img_instructions_data.py for info on the info.json file
        manuals = info["data"]["manuals"]

        for i in tqdm(range(len(manuals))):
            sample_tokens = []
            for j in range(len(manuals[i])):
                manual = manuals[i][j]
                instruction_tokens = []
                for instruction in manual['instructions']:
                    instruction_caption = f"{instruction[2][0]}{instruction[2][1]}"
                    # pad instruction caption
                    instruction_caption = instruction_caption + "*" * (max_caption_length - len(instruction_caption))
                    instruction_tokens.append(encode(instruction_caption))
                sample_tokens.append(instruction_tokens)
            tokens.append(sample_tokens)

    # TODO this only works if all samples have the same number of instructions
    # shape = (total_num_envs, num_samples_per_env, num_instructions_per_sample, max_caption_length)
    tokens = torch.Tensor(tokens).type(torch.uint8)
    torch.save(tokens, output_tokens_path)

if __name__ == "__main__":
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
    generate_single_step_tokens(
        data_folders=data_folders, 
        output_folder=output_folder,
    ) 