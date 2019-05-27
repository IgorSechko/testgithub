"""
Module for loading model from TF Object Detection Zoo

example:

python --name ssd_mobilenet_v2_coco_2018_03_29 --dir /content

"""
import urllib.request
import os
import tarfile
import argparse
import json


def download_model(dest_dir, model_name):
    print("Download pretrained model ...")
    if model_name is None:
        config = json.loads(open('../etc/config_json', 'r').read())
        frozen_model_path = os.path.join(config['root_dir'], 'frozen_model')
    else:
        frozen_model_path = os.path.join(dest_dir, 'frozen_model')

    if not os.path.exists(frozen_model_path):
        os.mkdir(frozen_model_path)

    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'

    urllib.request.urlretrieve(download_base + model_file, os.path.join(frozen_model_path, model_file))
    tar_file = tarfile.open(os.path.join(frozen_model_path, model_file))
    for file in tar_file.getmembers():
        tar_file.extract(file, frozen_model_path)
    print("Model saved to %s" % frozen_model_path)
