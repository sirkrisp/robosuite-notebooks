if __name__ == "__main__":
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("parent_dir", parent_dir)
    sys.path.append(str(parent_dir))

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as torch_data
from datagen.encode_images_dataset import EncodeDataset
from torchvision import models as torch_models
from torchvision import transforms
import numpy as np

DEVICE = "cuda"

# ImageNet statistics
DATA_MEANS = np.array([0.485, 0.456, 0.406])
DATA_STD = np.array([0.229, 0.224, 0.225])
# As torch tensors for later preprocessing
TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1,3,1,1)
TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1,3,1,1)


def encode_images():
    # setup paths
    data_folder = "/home/user/Documents/projects/robosuite-notebooks/datagen/clip_contrastive_01"
    images_filename = "keyframe_images.data"
    info_filename = "info.json"
    images_path = os.path.join(data_folder, images_filename)
    info_path = os.path.join(data_folder, info_filename)
    # output file path
    image_features_path = os.path.join(data_folder, "image_features_resnet.tar")

    # setup encoder
    print("setting up encoder...")
    resnet34_transform = transforms.Compose([transforms.Resize((224,224), antialias=True), transforms.Normalize(DATA_MEANS, DATA_STD)])
    model = torch_models.resnet34(weights='IMAGENET1K_V1')
    # Remove classification layer
    # In some models, it is called "fc", others have "classifier"
    # Setting both to an empty sequential represents an identity map of the final features.
    model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    model = model.to(DEVICE)
    # Only eval, no gradient required
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    def img_preprocess(img):
        img = np.moveaxis(img, -1, -3)
        img = torch.from_numpy(img / 255.0).float().to(DEVICE)
        return resnet34_transform(img)
    def encode_img_batch(img_batch):
        return model(img_batch)

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
        batch = batch.to(DEVICE)
        image_batch = encode_img_batch(batch)
        image_features[i:i+image_batch.shape[0]] = image_batch.detach().cpu()
        i += image_batch.shape[0]

    # save image features
    print("saving image features...")
    image_features = image_features.reshape((dataset.images_array_shape[0], dataset.images_array_shape[1], FEATURE_DIM))
    torch.save(image_features, image_features_path)

    print("done!")

if __name__ == "__main__":
    encode_images()


    




