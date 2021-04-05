import pandas as pd
import os
from scatter import get_dataset
import torch
from eval_ import visualize_indices_subplots

CONFIG_PATH = "configs/scan/scan_custom.yml"
CSV_NNS = "csvs/footwear_nearest.csv"
IMAGES_DIR = "/home/ITRANSITION.CORP/i.sechko/datasets/imaterialist/shoes_cropped/cutout_shoes/train"

IMAGE ="0bdd2a702fc6931da4699ba9415f959a_0_0.jpg" #"0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"# "ffc1e920e57417486bea549251492254_0_0.jpg" #"0b6741e4ea5ea7752803796d539fc214_0_1.jpg"# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"
# '18c22e4febd60c299968537bb35bbd2a_0_0.jpg'
# "0ac2aa7cb1299e97a5fa14df58e4cef7_0_0.jpg"
# "0b1351bc4e5942ac6a671636572d8a69_0_0.jpg"
# "0b6741e4ea5ea7752803796d539fc214_0_0.jpg"
# "0b6741e4ea5ea7752803796d539fc214_0_1.jpg"
# "00b10502fb082dcc8f156562b71f6f91_0_0.jpg"
# "0bdd2a702fc6931da4699ba9415f959a_0_0.jpg"
# "0c9c5d94960d1bfd7f4cdf97b0362213_0_0.jpg"
# "0d9e73a63eeefe78ff104ea8252a5818_0_1.jpg"
# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg"
# "ffc1e920e57417486bea549251492254_0_0.jpg"

image_names_os_sorted = torch.load("footwear_names_listdir.pt")
df = pd.read_csv(CSV_NNS, header=None)
dataset = get_dataset(CONFIG_PATH)
image_id = image_names_os_sorted.index(IMAGE)

indexes_to_show = df[df.iloc[:, 0] == image_id].iloc[0, 0:25].astype("int64").to_list()
dists = df[df.iloc[:, 0] == image_id].iloc[0, 25:50].astype("float64").to_list()
sum_ = df[df.iloc[:, 0] == image_id].iloc[0, 50].astype("float64")
avg = df[df.iloc[:, 0] == image_id].iloc[0, 51].astype("float64")
print(dists, sum_, avg)
print(indexes_to_show)
visualize_indices_subplots(indexes_to_show, dataset, "asd")

# "0ac2aa7cb1299e97a5fa14df58e4cef7_0_0.jpg" nice 0.1955563348531723
# "0b1351bc4e5942ac6a671636572d8a69_0_0.jpg" nice 0.3444426250457764
# "0b6741e4ea5ea7752803796d539fc214_0_0.jpg" idk, more than half irrelevant 0.6100771546363829
# "0b6741e4ea5ea7752803796d539fc214_0_1.jpg" tilted some green, nice 0.44657490372657777
# "00b10502fb082dcc8f156562b71f6f91_0_0.jpg" blue + many irrelevant 0.14447545766830444
# "0bdd2a702fc6931da4699ba9415f959a_0_0.jpg" nice white + all the same 0.3133584344387054 !!! take as target
# "0c9c5d94960d1bfd7f4cdf97b0362213_0_0.jpg" kros lies 0.4205123805999756
# "0d9e73a63eeefe78ff104ea8252a5818_0_1.jpg" ked 0.1509396493434906
# "0f36dfae21b87a5fe11dfb53b6d2e148_0_1.jpg" cool white 0.2535321080684662 !!!
# "ffc1e920e57417486bea549251492254_0_0.jpg" nice 0.4316281247138977
