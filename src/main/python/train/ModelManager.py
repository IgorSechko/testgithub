import os

from src.main.python.common.DataPreparator import DataPreparator
from src.main.python.common.FileUtils import clear_data_dir
from src.main.python.common.download_pretrained_model import download_model
from src.main.python.common.tensorflow.ssd_config import write_config

MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
ALLOWED_CATEGORIES = ['tee', 'dress']


class ModelManager:

    def __init__(self, allowed_category, model_name) -> None:
        self.root_dir = os.environ.get("AIBUY_TENSORFLOW_ROOT_DIR")
        self.data_dir = os.environ.get("AIBUY_TENSORFLOW_DATA_DIR")
        self.metadata_dir = os.path.join(self.data_dir, "metadata")
        self.allowed_category = allowed_category
        self.model_name = model_name

    def train(self):
        clear_data_dir(self.metadata_dir)
        self._data_preparation()
        # Start training
        download_model(dest_dir=self.data_dir, model_name=self.model_name)
        os.system(self._create_train_script())

    def _data_preparation(self):
        data_preparator = DataPreparator(self.data_dir, self.metadata_dir, self.allowed_category)
        data_preparator.build()
        write_config(self.model_name, self.data_dir, self.metadata_dir, data_preparator.get_category_count())

    def _create_train_script(self):
        export_path = "export PYTHONPATH=$PYTHONPATH:{0}/models/research/slim:{0}/models/research/".format(
            self.root_dir)
        train_script = "python {0}/models/research/object_detection/legacy/train.py".format(self.root_dir)
        log_error = "--logtostderr"
        pipeline_config = "--pipeline_config_path={0}/tf_api.config".format(self.metadata_dir)
        train_dir = "--train_dir={0}/checkpoints".format(self.metadata_dir)
        python_script = "{0}; {1} {2} {3} {4}".format(export_path, train_script, log_error, pipeline_config, train_dir)

        return python_script


if __name__ == '__main__':
    ModelManager(allowed_category=ALLOWED_CATEGORIES, model_name=MODEL_NAME).train()
