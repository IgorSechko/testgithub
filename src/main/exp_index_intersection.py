from annoy_utils import load_ann
from scatter import get_dataset
from eval_ import visualize_indices_subplots

TREE_FILES = [
    "annoy_trees/selflabel_imagenet200_footwear_euclidean_2048.ann",
    "annoy_trees/selflabel_imagenet100_footwear_euclidean_2048.ann",
    "annoy_trees/scan_imagenet200_footwear_euclidean_2048.ann",
    "annoy_trees/scan_imagenet100_footwear_euclidean_2048.ann",
    # "annoy_trees/scan_cifar10_footwear_euclidean_512.ann"
]

IMG_ID = 7362
NNS_NUM = 2000

if __name__ == "__main__":
    trees_list = []
    for tree_file in TREE_FILES:
        tree = load_ann(tree_file)
        trees_list.append(tree)

    trees_nns = []
    for tree in trees_list:
        nns = tree.get_nns_by_item(IMG_ID, NNS_NUM, include_distances=True)
        trees_nns.append(nns)

    nns_id_sets = [set(nns[0]) for nns in trees_nns]

    id_intersection = set.intersection(*nns_id_sets)
    id_intersection = list(id_intersection)


    def sort_rule(x):
        dists = []
        for tree in trees_nns:
            x_list_id = tree[0].index(x)
            dist = tree[1][x_list_id]
            dists.append(dist)
        avg = sum(dists) / len(dists)
        return avg


    id_intersection.sort(key=sort_rule)

    dataset = get_dataset("configs/scan/scan_custom.yml")

    visualize_indices_subplots(id_intersection[200:300], dataset)
