import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch

def merge_manuals(data_folders, output_folder, no_sep=False, merge_img_features=True, merge_img_features_finetuned=True, merge_captions_tokenized=True):
    filename = "captions_full" if not no_sep else "captions_full_no_sep"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # separate for loops for memory reasons

    if merge_img_features:
        print("Merging image features...")
        keyframe_features = []
        for data_folder in data_folders:
            keyframe_features.append(torch.load(f"{data_folder}/image_features.tar"))
        keyframe_features = torch.cat(keyframe_features, dim=0)
        torch.save(keyframe_features, f"{output_folder}/image_features.tar")
        print("Done merging image features")

    if merge_img_features_finetuned:
        print("Merging finetuned image features...")
        keyframe_features_finetuned = []
        for data_folder in data_folders:
            keyframe_features_finetuned.append(torch.load(f"{data_folder}/image_features_finetuned.tar"))
        keyframe_features_finetuned = torch.cat(keyframe_features_finetuned, dim=0)
        torch.save(keyframe_features_finetuned, f"{output_folder}/image_features_finetuned.tar")
        print("Done merging finetuned image features")  
        
    if merge_captions_tokenized:
        print("Merging tokenized captions...")
        captions_full_tokenized = []
        for data_folder in data_folders:
            captions_full_tokenized.append(torch.load(f"{data_folder}/{filename}_tokenized.tar"))
        captions_full_tokenized = torch.cat(captions_full_tokenized, dim=0)
        torch.save(captions_full_tokenized, f"{output_folder}/{filename}_tokenized.tar")
        print("Done merging tokenized captions")
        

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
    merge_manuals(
        data_folders=data_folders, 
        output_folder=output_folder, 
        no_sep=True,
        merge_img_features=False,
        merge_img_features_finetuned=True,
        merge_captions_tokenized=False,
    )