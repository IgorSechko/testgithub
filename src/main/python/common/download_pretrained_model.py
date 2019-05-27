import os
import tarfile
import urllib.request


def download_model(dest_dir, model_name):
    print("Download pretrained model ...")
    frozen_model_path = os.path.join(dest_dir, 'frozen_model')
    model_file = model_name + '.tar.gz'

    if os.path.exists(os.path.join(frozen_model_path, model_file)):
        print("Model already loaded.")
        return

    if not os.path.exists(frozen_model_path):
        os.mkdir(frozen_model_path)

    download_base = 'http://download.tensorflow.org/models/object_detection/'

    urllib.request.urlretrieve(download_base + model_file, os.path.join(frozen_model_path, model_file))
    tar_file = tarfile.open(os.path.join(frozen_model_path, model_file))
    for file in tar_file.getmembers():
        tar_file.extract(file, frozen_model_path)
    print("Model saved to %s" % frozen_model_path)
