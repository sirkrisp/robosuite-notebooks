import sys
# add path to osil
sys.path.append("/home/user/Documents/projects/osil")
# add path to this lib
sys.path.append("/home/user/Documents/projects/robosuite-notebooks")
from projects.clip_head.clip_head_model import CLIPHeadImgPairsPL

import torch
from tqdm import tqdm

DEVICE = "cuda"


def encode_images_finetuned(datafolders: str):
    # load model
    print("loading model...")
    run_id = "4r7zhy7c"
    ckpt_dir = f"/home/user/Documents/projects/osil/clip_head/{run_id}/checkpoints"
    ckpt_path = f"{ckpt_dir}/epoch=1120-step=78470.ckpt"
    clip_head_model = CLIPHeadImgPairsPL.load_from_checkpoint(ckpt_path)
    encoder = clip_head_model.model
    encoder.eval()
    encoder = encoder.to(DEVICE)

    for datafolder in datafolders:
        # load data
        print(f"loading data from {datafolder}...")
        image_features_path = f"{datafolder}/image_features.tar"
        image_features_finetuned_path = f"{datafolder}/image_features_finetuned.tar"
        image_features = torch.load(image_features_path)
        orig_shape = image_features.shape
        image_features = image_features.reshape(-1, orig_shape[-1])
        image_features_finetuned = torch.empty_like(image_features)

        # encode images
        batch_size = 10
        for i in tqdm(range(0, image_features.shape[0], batch_size)):
            image_features_batch = image_features[i:i+batch_size]
            b = image_features_batch.shape[0]
            image_features_batch = image_features_batch.to(DEVICE)
            image_features_finetuned_batch = encoder(image_features_batch).detach().cpu()
            image_features_finetuned[i:i+b] = image_features_finetuned_batch
        image_features_finetuned = image_features_finetuned.reshape(orig_shape)

        # save image features finetuned
        print("saving image features finetuned...")
        torch.save(image_features_finetuned, image_features_finetuned_path)

    print("done!")

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
    encode_images_finetuned(data_folders)