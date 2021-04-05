import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
import matplotlib
from tqdm import tqdm

from scatter import get_dataset

matplotlib.use("Agg")

EXPS_DIR = "/home/ITRANSITION.CORP/i.sechko/datasets/imaterialist/shoes_cropped/cutout_shoes/experiments/adt_123"


def save_to_subplots(img_name, indices, dataset):
    side = math.ceil(math.sqrt(len(indices)))
    if side == 1:
        side = 2

    fig, ax = plt.subplots(side, side)
    for idx, img_id in enumerate(indices):
        img = np.array(dataset.get_image(img_id)).astype(np.uint8)
        img = Image.fromarray(img)

        ax[idx // side, idx % side].imshow(img)
        ax[idx // side, idx % side].axis("off")

    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    # plt.show()
    plt.savefig(fname=img_name)
    plt.close()


def save_layer(layer_dir, layer, dataset):
    iter_ids = list(range(0, len(layer), 100))
    iter_ids.append(len(layer))

    for i in range(len(iter_ids) - 1):
        sample = layer[iter_ids[i]:iter_ids[i + 1]]
        img_name = f"{i}.png"
        img_path = os.path.join(layer_dir, img_name)
        save_to_subplots(img_path, sample, dataset)


def save_layers(exp_name, layers, dataset):
    exp_dir = os.path.join(EXPS_DIR, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_name)
    else:
        print("exp with the given name already exists.")
        return

    for i, layer in enumerate(layers):
        layer_dir = os.path.join(exp_dir, f"layer_{i}")
        os.makedirs(layer_dir)
        save_layer(layer_dir, layer, dataset)


if __name__ == "__main__":
    import torch
    import numpy as np

    dataset = get_dataset("configs/scan/scan_custom.yml")
    ans = torch.load("tensors/adaptive_32497.pt")
    layers_exps = ans["output"][1]
    thresholds = np.array(ans["output"][0])

    for i, layers_exp in tqdm(enumerate(layers_exps[1:6], start=1)):
        name = f"exp_{thresholds[i, 0]:.4f}_{int(thresholds[i, 1])}"
        save_layers(name, layers_exp, dataset)
