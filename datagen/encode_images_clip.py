if __name__ == "__main__":
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("parent_dir", parent_dir)
    sys.path.append(str(parent_dir))

import os
import torch
import open_clip
from tqdm import tqdm
import torch.utils.data as torch_data
from datagen.encode_images_dataset import EncodeDataset
from PIL import Image
import json
import numpy as np

device = "cuda"

def setup_endoer():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    def img_preprocess(img):
        img = Image.fromarray(img)
        return preprocess_val(img)
    model = model.to(device)
    model.eval()
    return model, img_preprocess

def encode_images(data_folders: list[str]):
    
    # setup encoder
    print("setting up encoder...")
    model, img_preprocess = setup_endoer()

    for data_folder in data_folders:
        print("data_folder", data_folder)
        # setup paths
        # data_folder = "/home/user/Documents/projects/robosuite-notebooks/datagen/manuals_04"
        images_filename = "keyframe_images.data"
        info_filename = "info.json"
        images_path = os.path.join(data_folder, images_filename)
        info_path = os.path.join(data_folder, info_filename)
        # output file path
        image_features_path = os.path.join(data_folder, "image_features.tar")

        # setup dataloader
        print("setting up dataloader...")
        batch_size = 10
        dataset = EncodeDataset(info_path, images_path, img_preprocess_fun=img_preprocess, preprocess=True)
        data_loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # , num_workers=4

        # encode images
        print("encoding images...")
        FEATURE_DIM = 512    
        image_features = torch.empty((len(dataset), FEATURE_DIM))
        i = 0
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            image_batch = model.encode_image(batch)
            image_features[i:i+image_batch.shape[0]] = image_batch.detach().cpu()
            i += image_batch.shape[0]

        # save image features
        print("saving image features...")
        image_features = image_features.reshape((*dataset.images_array_shape[:-3], FEATURE_DIM))
        torch.save(image_features, image_features_path)

        print("done!")

def images_to_tokens(data_folders: str, output_folder: str):

    # setup encoder
    print("setting up encoder...")
    model, img_preprocess = setup_endoer()
    model.visual.output_tokens = True

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
        n_keyframes = images_array_shape[2]

        # load images
        images = np.array(np.memmap(
            images_path,
            dtype=np.uint8,
            mode='r',
            shape=tuple(images_array_shape)
        )[:])
        images = np.reshape(images, (-1, n_keyframes, *images_array_shape[-3:]))  # (n_envs * n_samples_per_env, n_keyframes, *img_res, n_channels)

        for i in tqdm(range(images.shape[0])):
            preprocessed_keyframes = []
            for k in range(n_keyframes):
                img_preprocessed = img_preprocess(images[i][k]).to(device)
                preprocessed_keyframes.append(img_preprocessed)
            preprocessed_keyframes = torch.stack(preprocessed_keyframes)
            _, image_tokens  = model.encode_image(preprocessed_keyframes)
            image_tokens = image_tokens.detach().cpu()
            # save image tokens
            image_tokens_path = os.path.join(output_folder, f"image_tokens_{global_i}.tar")
            torch.save(image_tokens, image_tokens_path)
            global_i += 1


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
    encode_images(data_folders)
    # NOTE We do not have enough space to store all image features on the server.
    # images_to_tokens(data_folders=data_folders, output_folder="/home/user/Documents/projects/robosuite-notebooks/datagen/generated/manuals_01-10/image_tokens")


    




