from annoy import AnnoyIndex
import os
import torch

TENSOR = "tensors/cifar10_features.pt"


def build_annoy_trees(prefix_name, features, distances="all"):
    if distances == "all":
        distances = ["angular", "euclidean", "manhattan", "hamming", "dot"]

    vectors_num = features.shape[0]
    features_num = features.shape[1]

    for distance in distances:
        t = AnnoyIndex(features_num, distance)
        for i in range(vectors_num):
            v = features[i].tolist()
            t.add_item(i, v)

        t.build(100)
        filename = f"annoy_trees/{prefix_name}_{distance}_{features_num}.ann"
        t.save(filename)


def load_ann(filepath):
    filename = os.path.split(filepath)[-1]
    filename_no_ext = filename[:-4]
    distance, features_num = filename_no_ext.split("_")[-2:]
    features_num = int(features_num)

    u = AnnoyIndex(features_num, distance)
    u.load(filepath)

    return u


if __name__ == "__main__":
    features = torch.load(TENSOR)
    build_annoy_trees("scan_cifar10_footwear", features, distances=["euclidean"])
