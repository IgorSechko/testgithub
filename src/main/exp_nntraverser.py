import os

from tqdm import tqdm
import pandas as pd
import annoy_utils
import torch
import numpy as np
import shutil
from scatter import get_dataset

# from save_layers import save_layers

# _, features = torch.load("tensors/selflabel_imagenet_200_footwear_output.pt")
# annoy_utils.build_annoy_trees("selflabel_imagenet200_footwear", features=features, distances=["euclidean"])

# tree = annoy_utils.load_ann("annoy_trees/selflabel_imagenet200_footwear_euclidean_2048.ann")
# vectors_num = tree.get_n_items()

# for idx in tqdm(range(vectors_num)):
#     indexes, dists = tree.get_nns_by_item(idx, 25, include_distances=True)
#     dists_sum = sum(dists)
#     dists_avg = dists_sum / len(dists)
#
#     row = indexes + dists + [dists_sum, dists_avg]
#     csv_row = ",".join(map(str, row)) + "\n"
#
#     with open("csvs/footwear_nearest.csv", "a+") as f:
#         f.write(csv_row)

# "0ac2aa7cb1299e97a5fa14df58e4cef7_0_0.jpg"
# "0b1351bc4e5942ac6a671636572d8a69_0_0.jpg"
# "0b6741e4ea5ea7752803796d539fc214_0_0.jpg"
# "0b6741e4ea5ea7752803796d539fc214_0_1.jpg"
# "00b10502fb082dcc8f156562b71f6f91_0_0.jpg"
# "0bdd2a702fc6931da4699ba9415f959a_0_0.jpg" nice white + all the same 0.3133584344387054 !!! take as target
# "0c9c5d94960d1bfd7f4cdf97b0362213_0_0.jpg"
# "0d9e73a63eeefe78ff104ea8252a5818_0_1.jpg"
# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"

IMAGE = "0bdd2a702fc6931da4699ba9415f959a_0_0.jpg"  # "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"  # '0ac2aa7cb1299e97a5fa14df58e4cef7_0_1.jpg'  # "03732d448d7d84134c8a99a61a61670f_0_1.jpg"
IMAGES_DIR = "/home/ITRANSITION.CORP/i.sechko/datasets/imaterialist/shoes_cropped/cutout_shoes/train"
DEST_DIR = "/home/ITRANSITION.CORP/i.sechko/datasets/imaterialist/shoes_cropped/cutout_shoes/test_1"
THRESHOLD = 0.35
MAX_NNS_RETURN = 100


class NNTraverser:

    def __init__(self, annoy_index):
        self.annoy_index = annoy_index
        self.items_num = annoy_index.get_n_items()
        self.ids_seen = np.zeros(self.items_num, dtype=np.bool)
        self.ids_times = np.zeros(self.items_num, dtype=np.int32)
        self.depths = None  # np.zeros(1000, dtype=np.int32)

    def get_nns_within(self, index, threshold):
        all_ids, all_dists = self.annoy_index.get_nns_by_item(index, MAX_NNS_RETURN, include_distances=True)

        all_ids = np.array(all_ids, dtype=np.int32)
        all_dists = np.array(all_dists, dtype=np.float64)

        dists_within_t = all_dists[np.where(all_dists < threshold)]
        ids_within_t = all_ids[np.where(all_dists < threshold)]

        return ids_within_t, dists_within_t

    def get_unseen_nns_within(self, index, threshold):
        ids, dists = self.get_nns_within(index, threshold)

        unseen_mask = self.get_unseen_mask(ids)
        ids = ids[unseen_mask]
        dists = dists[unseen_mask]

        return ids, dists

    def get_unseen_mask(self, indexes):
        seen_mask = self.ids_seen[indexes]
        unseen_mask = np.invert(seen_mask)
        return unseen_mask

    @staticmethod
    def get_edges(index, unseen_indexes):
        edges = [(index, unseen) for unseen in unseen_indexes]
        return edges

    def set_seen(self, indexes):
        self.ids_seen[indexes] = True

    def set_all_unseen(self):
        self.ids_seen = np.zeros(self.items_num, dtype=np.bool)

    def get_next_layer(self, in_layer_ids, threshold, tree_mapping):
        out_layer_ids = []
        out_layer_dists = []

        for index in in_layer_ids:
            ids, dists = self.get_unseen_nns_within(index, threshold)
            self.set_seen(ids)
            tree_mapping[index] = (ids, dists)
            out_layer_ids.append(ids)
            out_layer_dists.append(dists)

        out_layer_ids = np.concatenate(out_layer_ids)
        out_layer_dists = np.concatenate(out_layer_dists)

        return out_layer_ids, out_layer_dists

    def set_next_layer(self, in_layer_ids, threshold, mapping):
        for index in in_layer_ids:
            ids, dists = self.get_unseen_nns_within(index, threshold)
            self.set_seen(ids)
            mapping[index] = (ids, dists)

    def traverse_breadth_first(self, target_id, threshold, layer_num_limit=1000000,
                               threshold_cutoff=0.0):
        self.set_all_unseen()
        in_layer_ids = np.array([target_id])
        in_layer_dists = np.array([0])
        self.set_seen(in_layer_ids)

        threshold_cutoff = threshold_cutoff * threshold
        id_layers = []
        dist_layers = []
        thresholds = []
        layer_num = 1
        tree_mapping = [None] * self.items_num

        thresholds.append(threshold)
        id_layers.append(in_layer_ids)
        dist_layers.append(in_layer_dists)
        while in_layer_ids.shape[0] > 0 and layer_num <= layer_num_limit:
            out_layer_ids, out_layer_dists = self.get_next_layer(in_layer_ids, threshold, tree_mapping)
            in_layer_ids = out_layer_ids
            in_layer_dists = out_layer_dists
            id_layers.append(in_layer_ids)
            dist_layers.append(in_layer_dists)
            layer_num += 1

            threshold = threshold - threshold_cutoff
            thresholds.append(threshold)

        return id_layers, dist_layers, thresholds, tree_mapping

    def traverse_depth_first(self, index, threshold, depth_limit):
        self.set_all_unseen()
        self.set_seen(index)
        self.depths = [[] for _ in range(depth_limit)]
        self.depths[0].append(np.array([index]))
        self.traverse_recursively(index, threshold, depth_limit=depth_limit)
        ans = []
        for depth in self.depths:
            ans.append(np.concatenate(depth))
        return ans

    def traverse_recursively(self, index, threshold, depth=1, depth_limit=None):
        if depth == depth_limit:
            return
        ids, dists = self.get_unseen_nns_within(index, threshold)
        self.depths[depth].append(ids)
        self.set_seen(ids)
        self.ids_times[ids] += 1
        for idx in ids:
            self.traverse_recursively(idx, threshold, depth=depth + 1, depth_limit=depth_limit)


def copy_images(image_names):
    for image_name in image_names:
        src = os.path.join(IMAGES_DIR, image_name)
        dst = os.path.join(DEST_DIR, image_name)
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    trees = annoy_utils.load_ann("annoy_trees/selflabel_imagenet200_footwear_euclidean_2048.ann")
    image_names_os_sorted = torch.load("footwear_names_listdir.pt")
    # print("image id:", image_id)

    dataset = get_dataset("configs/scan/scan_custom.yml")

    image_id = image_names_os_sorted.index(IMAGE)
    traverser = NNTraverser(trees)
    layers, layer_dists, _, tree_mapping = traverser.traverse_breadth_first(image_id, THRESHOLD)

    info = {
        "target_id": image_id,
        "threshold": THRESHOLD,
        "tree_mapping": tree_mapping
    }

    import torch
    torch.save(info, f"tensors/tree_mapping_{image_id}_{THRESHOLD:.2f}.pt")
    ################################################################################
    ################################################################################
    # elems, dists = trees.get_nns_by_item(image_id, 45127, include_distances=True)
    #
    # import matplotlib.pyplot as plt
    # plt.hist(dists, bins=2000)
    # plt.show()

    # items_num = tree.get_n_items()
    # ids_seen = np.array([False] * items_num)
    # ans = get_unseen_nns_within(12, 100)
    # id_times = np.array([0] * items_num)
    # depths = [0] * 1000
    # copy_images(np.array(image_names_os_sorted)[id_seen])

    # ans = traverser.traverse_depth_first(image_id, THRESHOLD, depth_limit=5)
    # for i, threshold in enumerate(np.linspace(start=0.16156942943529962, stop=0.16156943643529962, num=10000)):
    #     traverser = NNTraverser(trees)
    #     layers_ids, layers_dists = traverser.traverse_breadth_first(image_id, threshold)
    #     print("threshold:", threshold)
    #     print("all neighbors found:", traverser.ids_seen.nonzero()[0].__len__())
    #     print("iteration:", i)
    #     del traverser
    ################################################################################
    ################################################################################

    #############################################################
    #############################################################
    # image_id = image_names_os_sorted.index(IMAGE)
    # cutoff_num_elems = []
    # cutoff_layers = []
    # threshold_cutoffs = np.linspace(start=0.0, stop=0.30, num=20)
    # for threshold_cutoff in threshold_cutoffs:
    #
    #     traverser = NNTraverser(trees)
    #     output = traverser.traverse_breadth_first(image_id, THRESHOLD, 100,
    #                                               threshold_cutoff)
    #     layers_ids, layers_dists, edges, thresholds = output
    #     # save_layers(f"1_ad_thr_{threshold_cutoff}", layers_ids, dataset)
    #     print("\n########################################################")
    #     print(f"Image: {IMAGE}")
    #     print(f"Image id: {image_id}")
    #     print("threshold cutoff:", threshold_cutoff)
    #     all_elems_num = traverser.ids_seen.nonzero()[0].__len__()
    #     print("all cluster elements:", all_elems_num)
    #     for i, layer in enumerate(layers_ids):
    #         print("layer", i, "contains", len(layer), "elements")
    #         print(f"threshold: {thresholds[i]:.4f}", )
    #
    #     cutoff_num_elems.append([threshold_cutoff, all_elems_num])
    #     cutoff_layers.append(layers_ids)
    #
    #     del traverser
    #
    # import matplotlib.pyplot as plt
    #
    # arr = np.array(cutoff_num_elems)
    # plt.plot(arr[:, 0], arr[:, 1])
    # plt.xticks(threshold_cutoffs)
    # plt.yticks(np.linspace(0, 45000, 20))
    # plt.xlabel("part of initial threshold subtracted with every layer")
    # plt.ylabel("all elems on all levels")
    # plt.grid()
    # plt.show()
    # plt.savefig(f"visualized/ad_thr/img_id_{image_id}.png")
    #
    # import torch
    #
    # info = {
    #     "image": IMAGE,
    #     "image_id": image_id,
    #     "init_threshold": THRESHOLD,
    #     "output": (cutoff_num_elems, cutoff_layers)
    # }
    # torch.save(info, f"tensors/adaptive_{image_id}.pt")
    ################################################################################
    ################################################################################

# '0ac2aa7cb1299e97a5fa14df58e4cef7_0_1.jpg'
# [0.0, 0.1259140968322754, 0.16156943142414093, 0.16310960054397586, 0.16404679417610168, 0.16492824256420135, 0.1650322675704956, 0.16773584485054016, 0.16788533329963684, 0.16966761648654938, 0.17274659872055054, 0.17338106036186218, 0.179043710231781, 0.17936913669109344, 0.179631769657135, 0.1810285598039627, 0.1811835914850235, 0.18157367408275604, 0.18182161450386047, 0.18182408809661865, 0.18233457207679749, 0.1826434135437012, 0.18287086486816406, 0.18304261565208435, 0.1833563894033432] 4.155740886926652 0.16622963547706604
# [19503, 10179, 9255, 23279, 39945, 6230, 29676, 44974, 27456, 41010, 8297, 32322, 40303, 459, 11067, 11239, 23296, 43045, 8903, 42475, 20384, 17261, 30833, 696, 36961]
#


# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"
# [0.0, 0.2199556529521942, 0.22052700817584991, 0.23200614750385284, 0.25727495551109314, 0.2578832507133484, 0.2607994377613068, 0.2614578902721405, 0.2615413069725037, 0.26468172669410706, 0.2653825879096985, 0.26612117886543274, 0.2667833864688873, 0.2701994180679321, 0.2707981169223785, 0.2734460830688477, 0.2737222909927368, 0.2737841010093689, 0.2750918567180633, 0.27560022473335266, 0.2779049277305603, 0.2781407237052917, 0.27830785512924194, 0.2783159613609314, 0.2785766124725342] 6.3383027017116556 0.2535321080684662
# [21997, 8731, 1392, 3939, 43842, 27855, 37401, 25230, 27499, 17474, 36722, 31902, 17321, 15091, 19806, 39901, 15187, 28699, 3919, 25412, 14226, 44512, 35886, 21379, 35404]
# threshold: 0.23200614750385284
# all neighbors found: 3
# iteration: 549
# threshold: 0.23200614750385287
# all neighbors found: 28434
# iteration: 550

# '18c22e4febd60c299968537bb35bbd2a_0_0.jpg'
# [0.0, 0.21932369470596316, 0.23200614750385284, 0.23898573219776156, 0.2401948720216751, 0.24456606805324554, 0.24732309579849246, 0.2547028362751007, 0.25596725940704346, 0.2565707564353943, 0.25970572233200073, 0.2662887275218964, 0.2695014476776123, 0.2716517448425293, 0.2729508876800537, 0.2737781703472137, 0.2742907702922821, 0.27479952573776245, 0.2748095691204071, 0.2756598889827728, 0.2759280800819397, 0.27727723121643066, 0.277293860912323, 0.2781126499176025, 0.2784828245639801] 6.290171563625336 0.2516068625450134
# [3939, 816, 21997, 17321, 44512, 15187, 1392, 10690, 31290, 28615, 36722, 23261, 40779, 19491, 30609, 43842, 12619, 6508, 33805, 26244, 2739, 36744, 939, 27499, 2644]
# threshold: 0.21932369470596313
# all neighbors found: 1
# iteration: 268
# threshold: 0.21932369470596316
# all neighbors found: 27198
# iteration: 269

# "0ac2aa7cb1299e97a5fa14df58e4cef7_0_0.jpg"
# [0.0, 0.16773584485054016, 0.1750076413154602, 0.17809580266475675, 0.1804673373699188, 0.18594151735305786, 0.19382257759571075, 0.1949583888053894, 0.19677165150642395, 0.2066633254289627, 0.2080222964286804, 0.20909613370895386, 0.21004445850849152, 0.21022020280361176, 0.21271200478076932, 0.2131513357162476, 0.21334657073020932, 0.2134014368057251, 0.21340391039848328, 0.21638651192188266, 0.21660244464874268, 0.2178776115179062, 0.21828493475914, 0.2183040827512741, 0.21859034895896912] 4.8889083713293084 0.1955563348531723
# [44974, 19503, 10179, 17147, 13516, 6230, 30523, 16149, 16531, 24049, 25388, 39945, 12202, 16958, 30448, 1073, 3242, 29250, 39583, 12463, 37460, 16203, 9860, 3605, 30833]
# threshold: 0.16773584485054016
# number found: 1
# iteration: 4363
# threshold: 0.16773584485054022
# number found: 20383
# iteration: 4364

# "ffc1e920e57417486bea549251492254_0_0.jpg"
# [0.0, 0.4251948595046997, 0.4291194975376129, 0.4308747351169586, 0.43551549315452576, 0.4366378486156464, 0.4380339682102203, 0.43999505043029785, 0.4432097375392914, 0.4444378316402435, 0.4460790455341339, 0.4499093890190125, 0.4507391154766083, 0.4530618190765381, 0.4539511501789093, 0.4583093225955963, 0.4586889147758484, 0.4596734642982483, 0.4600447416305542, 0.4620381593704224, 0.4623565673828125, 0.4624519050121307, 0.4629846513271332, 0.4633376002311706, 0.4640582501888275] 10.790703117847443 0.4316281247138977
# [44377, 36671, 25840, 18778, 40834, 43329, 41900, 20613, 30541, 29192, 15080, 3571, 17269, 18147, 3234, 29738, 43735, 19565, 28791, 31118, 35313, 9638, 37748, 12828, 25603]
# threshold: 0.425194859504-68655
# all neighbors found: 1
# iteration: 1531
# threshold: 0.425194859504-70653
# all neighbors found: 42027
# iteration: 1532

# '0ac2aa7cb1299e97a5fa14df58e4cef7_0_1.jpg'
# [0.0, 0.1259140968322754, 0.16156943142414093, 0.16310960054397586, 0.16404679417610168, 0.16492824256420135, 0.1650322675704956, 0.16773584485054016, 0.16788533329963684, 0.16966761648654938, 0.17274659872055054, 0.17338106036186218, 0.179043710231781, 0.17936913669109344, 0.179631769657135, 0.1810285598039627, 0.1811835914850235, 0.18157367408275604, 0.18182161450386047, 0.18182408809661865, 0.18233457207679749, 0.1826434135437012, 0.18287086486816406, 0.18304261565208435, 0.1833563894033432] 4.155740886926652 0.16622963547706604
# [19503, 10179, 9255, 23279, 39945, 6230, 29676, 44974, 27456, 41010, 8297, 32322, 40303, 459, 11067, 11239, 23296, 43045, 8903, 42475, 20384, 17261, 30833, 696, 36961]
# threshold: 0.16156943142-349844
# all neighbors found: 3
# iteration: 2840
# threshold: 0.16156943142-41985
# all neighbors found: 19187
# iteration: 2841
