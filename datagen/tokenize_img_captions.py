import os
import torch
from tqdm import tqdm


def tokenize_img_captions(data_folder: str, max_length: int = None, no_sep=False):
    # data_folder = os.path.join(os.path.dirname(__file__), "manuals_01")
    filename = "captions_full" if not no_sep else "captions_full_no_sep"
    captions_path = os.path.join(data_folder, f"{filename}.txt")
    captions_tokenized_path = os.path.join(data_folder, f"{filename}_tokenized.tar")

    # load captions
    with open(captions_path, 'r') as f:
        captions = f.read()
    captions = captions.replace(" ", "").split("\n")
    captions = [caption for caption in captions if caption != ""]

    # create encoder
    # chars = ["A", "B", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", "(", ")", "{", "}", ",", ".", ":", "*"]
    chars = ["A", "B", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "*"] if no_sep else ["A", "B", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", "(", ")", "{", "}", ",", ".", ":", "*"]
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")
    stoi = { ch:i for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    captions_tokenized = []
    for caption in tqdm(captions):
        caption_tokenized = encode(caption)
        captions_tokenized.append(caption_tokenized)

    # padding to the same length
    max_length_ = max([len(caption_tokenized) for caption_tokenized in captions_tokenized])
    if max_length is None:
        max_length = max_length_
    else:
        assert max_length_ <= max_length, f"max_length should be >= {max_length_}"
    print("max_length", max_length, "max_length_", max_length_)
    for i in tqdm(range(len(captions_tokenized))):
        captions_tokenized[i] = captions_tokenized[i] + [stoi["*"]] * (max_length - len(captions_tokenized[i]))
    
    torch.save(torch.Tensor(captions_tokenized).type(torch.uint8), captions_tokenized_path)

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
        # tokenize_img_captions(data_folder, max_length=148, no_sep=True)
        tokenize_img_captions(data_folder, max_length=46, no_sep=True)

    