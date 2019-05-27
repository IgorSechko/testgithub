import os

from src.main.python.common.CommonUtils import load_config
from src.main.python.common.FileUtils import clear_data_dir
from src.main.python.common.data_preparator import DataPreparator
from src.main.python.common.download_tf_zoo_model import download_model
from src.main.python.common.ssd_config import write_config

MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
ALLOWED_CATEGORIES = ['blazer', 'blouse', 'cardigan', 'hoodie', 'jacket', 'sweater', 'tank', 'tee',
                      'top', 'jeans', 'joggers', 'leggings', 'shorts', 'skirt', 'sweatpants', 'coat',
                      'dress', 'jumpsuit', 'kimono', 'romper']

class ModelManager:

    def __init__(self, allowed_category, model_name) -> None:
        config = load_config()
        self.root_dir = config['root_dir']
        self.allowed_category = allowed_category
        self.model_name = model_name

    def train(self):
        clear_data_dir("{0}/data_dir/".format(self.root_dir))
        self._data_preparation()
        # Start training
        download_model(dest_dir=self.root_dir, model_name=self.model_name)
        os.system(self._create_train_script())

    def _data_preparation(self):
        # API_PATH = os.path.join(self.root_dir, 'models/research')
        # sys.path.append(API_PATH)
        #
        # DETECTOR_PATH = os.path.join(self.root_dir, 'TFFashionDetection')
        # sys.path.append(DETECTOR_PATH)

        data_preparator = DataPreparator(self.allowed_category)
        data_preparator.build()
        write_config(self.model_name, self.root_dir, data_preparator.get_category_count())

    def _create_train_script(self):
        export_path = "export PYTHONPATH=$PYTHONPATH:{0}/models/research/slim:{0}/models/research/".format(
            self.root_dir)
        train_script = "python {0}/models/research/object_detection/legacy/train.py".format(self.root_dir)
        log_error = "--logtostderr"
        pipeline_config = "--pipeline_config_path={0}/data_dir/tf_api.config".format(self.root_dir)
        train_dir = "--train_dir={0}/data_dir/checkpoints".format(self.root_dir)
        python_script = "{0}; {1} {2} {3} {4}".format(export_path, train_script, log_error, pipeline_config, train_dir)

        return python_script


if __name__ == '__main__':
    ModelManager(allowed_category=ALLOWED_CATEGORIES, model_name=MODEL_NAME).train()
