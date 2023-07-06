
from torchvision.datasets import CIFAR100
import torch
import torch.nn as nn
import numpy as np

from torchvision import models as torch_models
from torchvision import transforms

import os
import json
from tqdm import tqdm


# ImageNet statistics
DATA_MEANS = np.array([0.485, 0.456, 0.406])
DATA_STD = np.array([0.229, 0.224, 0.225])
# As torch tensors for later preprocessing
TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1,3,1,1)
TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1,3,1,1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":

    dataset_folder = "/home/user/Documents/projects/robosuite-notebooks/datagen/keyframes_merged_5_7_8_9"

    resnet34_transform = transforms.Compose([transforms.Resize((224,224)), transforms.Normalize(DATA_MEANS, DATA_STD)])
    resnet34 = torch_models.resnet34(weights='IMAGENET1K_V1')
    # Remove classification layer
    # In some models, it is called "fc", others have "classifier"
    # Setting both to an empty sequential represents an identity map of the final features.
    resnet34.fc = nn.Sequential()
    resnet34.classifier = nn.Sequential()
    # To GPU
    resnet34 = resnet34.to(DEVICE)

    # Only eval, no gradient required
    resnet34.eval()
    for p in resnet34.parameters():
        p.requires_grad = False


    info = json.load(open(os.path.join(dataset_folder, "info.json"), "r"))

    total_num_images = info["dataset"]["total_num_images"]
    print("total_num_images: ", total_num_images)

    # Load keyframes
    img_res = (300, 300)
    images = np.memmap(
            os.path.join(dataset_folder, "keyframe_images.data"),
            dtype=np.uint8,
            mode='r',
            shape=(total_num_images, 3, *img_res)
        )[:]

    images = images
    # images = np.moveaxis(images, -1, -3)  # shape (total_num_images, 3, 300, 300)


    image_features = torch.zeros((total_num_images, 512), dtype=torch.float32).to(DEVICE)

    batch_size = 100
    num_batches = total_num_images // batch_size
    assert num_batches * batch_size == total_num_images
    for i in tqdm(range(num_batches)):
        batch = torch.from_numpy(images[i*batch_size:(i+1)*batch_size, :] / 255.0).float().to(DEVICE)
        batch = resnet34_transform(batch)
        batch_features = resnet34(batch)
        image_features[i*batch_size:(i+1)*batch_size, :] = batch_features
    

    # Save image features
    image_features = image_features.detach().cpu()
    torch.save(image_features, os.path.join(dataset_folder, "image_features.tar"))
