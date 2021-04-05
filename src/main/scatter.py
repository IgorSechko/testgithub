import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from utils.common_config import get_val_dataset, get_val_transformations
from matplotlib.figure import figaspect

CONFIG_PATH = "configs/scan/scan_custom.yml"
# EMBEDDINGS_2D = "tensors/footwear_features_umap_embedded.pt"

DISPLAY_TARGETS = False
TARGETS_FILE = "tensors/cifar10_targets.pt"


def visualize_indices(img_index, dataset, point_num):
    img = np.array(dataset.get_image(img_index)).astype(np.uint8)
    h, w, _ = img.shape
    figsize = figaspect(h/w)
    img = Image.fromarray(img)
    fig_ = plt.figure()
    fig_title = f"Point_{point_num}"
    fig_.canvas.set_window_title(fig_title)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(img)
    # plt.margins(0,0)
    # plt.savefig(f"figures/footwear/{fig_title}")#, bbox_inches='tight',           pad_inches=0)
    plt.show()


def get_dataset(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)

    return dataset


def on_pick(event):
    first_index = event.ind[0]
    x = embeddings[first_index, 0]
    y = embeddings[first_index, 1]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.scatter(x, y, color="red")
    ax.text(x, y, f"{first_index}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # fig.canvas.draw()
    visualize_indices(first_index, dataset, first_index)


if __name__ == "__main__":
    dataset = get_dataset(CONFIG_PATH)

    embedding_files = [
        "tensors/scan_imagenet_100_footwear_output_pca_embedded.pt",
        "tensors/scan_imagenet_100_footwear_output_umap_embedded.pt",
        "tensors/scan_imagenet_200_footwear_output_pca_embedded.pt",
        "tensors/scan_imagenet_200_footwear_output_umap_embedded.pt",
        "tensors/selflabel_imagenet_100_footwear_output_pca_embedded.pt",
        "tensors/selflabel_imagenet_100_footwear_output_umap_embedded.pt",
        "tensors/selflabel_imagenet_200_footwear_output_pca_embedded.pt",
        "tensors/selflabel_imagenet_200_footwear_output_umap_embedded.pt"
    ]

    embeddings = torch.load(embedding_files[7])

    fig = plt.figure()
    ax = fig.add_subplot()

    targets = None
    if DISPLAY_TARGETS:
        targets = torch.load(TARGETS_FILE)

    line = ax.scatter(embeddings[:10000, 0], embeddings[:10000, 1], c=targets, label=targets, picker=True)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.show()
