import math
import os
import matplotlib.pyplot as plt

import torch
from PIL import Image
import numpy as np
import matplotlib

from scatter import get_dataset

matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 5})


class TreeVisualizer:
    def __init__(self, tree_mapping, target_id, dataset):
        self.tree_mapping = tree_mapping
        self.target_id = target_id
        self.dataset = dataset

    def visualize(self, root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        os.chdir(root_dir)
        img_name = os.path.join(os.getcwd(), f"target_{self.target_id}.png")
        self.save_sample_to_subplots(img_name, [self.target_id], [0.0])
        self.traverse_in_depth(self.target_id, 0.0)

    def traverse_in_depth(self, image_id, distance):
        # if image_id points at sth
        children_ids = self.tree_mapping[image_id][0].tolist()
        dists_to_children = self.tree_mapping[image_id][1].tolist()
        if children_ids:
            # create dir and cd
            dir_name = f"{image_id}_{distance:.10f}"
            os.makedirs(dir_name)
            os.chdir(dir_name)
            # put children photos into dir
            self.save_children_subplots([image_id] + children_ids, [0.0] + dists_to_children)

            for i in range(len(children_ids)):
                child_id = children_ids[i]
                dist_to_child = dists_to_children[i]
                self.traverse_in_depth(child_id, dist_to_child)

            os.chdir("../")

    def save_sample_to_subplots(self, img_name, indices, dists):
        side = math.ceil(math.sqrt(len(indices)))

        fig, ax = plt.subplots(side, side)
        if side == 1:
            img_id = indices[0]
            img = np.array(self.dataset.get_image(img_id)).astype(np.uint8)
            img = Image.fromarray(img)

            ax.imshow(img)
            ax.set_title(f"{indices[0]}", fontdict={"fontweight": 800})
            ax.axis("off")

        else:
            img_id = indices[0]
            idx = 0
            has_children = ""
            if self.tree_mapping[img_id][0].shape[0] == 0:
                has_children = "_no"
            img = np.array(self.dataset.get_image(img_id)).astype(np.uint8)
            img = Image.fromarray(img)

            ax[idx // side, idx % side].imshow(img)
            ax[idx // side, idx % side].set_title(f"{img_id}_PARENT" + has_children, fontdict={"fontweight": 800})
            ax[idx // side, idx % side].axis("off")

            for idx, img_id in enumerate(indices[1:], start=1):
                has_children = ""
                if self.tree_mapping[img_id][0].shape[0] == 0:
                    has_children = "_no"
                img = np.array(self.dataset.get_image(img_id)).astype(np.uint8)
                img = Image.fromarray(img)

                ax[idx // side, idx % side].imshow(img)
                ax[idx // side, idx % side].set_title(f"{img_id}_{dists[idx]:.6f}" + has_children)
                ax[idx // side, idx % side].axis("off")

        fig.tight_layout(w_pad=0.1, h_pad=0.1)
        # plt.show()
        plt.savefig(fname=img_name)
        plt.close()

    def save_children_subplots(self, parent_and_children, dists):
        iter_ids = list(range(0, len(parent_and_children), 100))
        iter_ids.append(len(parent_and_children))

        for i in range(len(iter_ids) - 1):
            sample_ids = parent_and_children[iter_ids[i]:iter_ids[i + 1]]
            sample_dists = dists[iter_ids[i]:iter_ids[i + 1]]
            img_name = f"{i}.png"
            img_path = os.path.join(os.getcwd(), img_name)
            self.save_sample_to_subplots(img_path, sample_ids, sample_dists)


if __name__ == "__main__":
    info = torch.load("tensors/tree_mapping_7362_0.35.pt")
    tree_mapping = info["tree_mapping"]

    dataset = get_dataset("configs/scan/scan_custom.yml")

    visualizer = TreeVisualizer(tree_mapping, 7362, dataset)
    root_dir = "/home/ITRANSITION.CORP/i.sechko/datasets/imaterialist/shoes_cropped/cutout_shoes/tree11131231"
    visualizer.visualize(root_dir)
