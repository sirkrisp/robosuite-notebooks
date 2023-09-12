
import numpy as np
import json
from PIL import Image
import torch.utils.data as torch_data

class EncodeDataset(torch_data.Dataset):

    def __init__(self, info_path, images_path, img_preprocess_fun, preprocess=True) -> None:
        super().__init__()

        # load info
        info = json.load(open(info_path, "r"))
        self.images_array_shape = info["data"]["images"]["shape"]

        # load images
        self.images = np.array(np.memmap(
            images_path,
            dtype=np.uint8,
            mode='r',
            shape=tuple(self.images_array_shape)
        )[:])
        self.images = np.reshape(self.images, (-1, *self.images_array_shape[-3:]))

        self.img_preprocess_fun = img_preprocess_fun
        self.preprocess = preprocess

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        img = self.images[index]
        if self.preprocess:
            img = self.img_preprocess_fun(img)
        return img