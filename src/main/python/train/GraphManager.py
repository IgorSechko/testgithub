import argparse
import datetime
import glob
import os
import re
import sys

from src.main.python.common.FileUtils import model_to_zip

EXPORT_PATH = "export PYTHONPATH=$PYTHONPATH:{0}/models/research/slim:{0}/models/research/"
EXPORT_MODEL = "python {0}/models/research/object_detection/export_inference_graph.py"
CHECKPOINT_MODEL = "{0}/data_dir/checkpoints/model.ckpt-{1}"
OUTPUT_DIRECTORY = "{0}/inference_graph/graph{1}_{2}"
TF_API_CONFIG = "{0}/data_dir/tf_api.config"
LABEL_FILE = "{0}/data_dir/annotations/label_map.pbtxt"

_PARAM_TRAINED_CHECKPOINT_PREFIX = "--trained_checkpoint_prefix=" + CHECKPOINT_MODEL
_PARAM_OUTPUT_DIRECTORY = "--output_directory " + OUTPUT_DIRECTORY
_PARAM_PIPELINE_CONFIG_PATH = "--pipeline_config_path=" + TF_API_CONFIG
_PARAM_INPUT_TYPE_IMAGE_TENSOR = "--input_type image_tensor"


class GraphManagement:

    def __init__(self) -> None:
        self.root_dir = os.environ.get("AIBUY_TENSORFLOW_ROOT_DIR")
        self.data_dir = os.environ.get("AIBUY_TENSORFLOW_DATA_DIR")

    def export(self, model_number):
        # Create graph from checkpoints
        try:
            os.system(self._create_export_script(model_number))

            self.export_model(model_number)
            print("Model was exported")
        except Exception as e:
            print("Process error: {0}".format(e))

    def _create_export_script(self, model_number):
        export_path = EXPORT_PATH.format(self.root_dir)

        python_export_model = EXPORT_MODEL.format(self.root_dir)
        pipeline_parameter = _PARAM_PIPELINE_CONFIG_PATH.format(self.data_dir)
        checkout_prefix = _PARAM_TRAINED_CHECKPOINT_PREFIX.format(self.data_dir, model_number)

        output_directory = _PARAM_OUTPUT_DIRECTORY.format(self.data_dir, model_number, self._get_current_date())

        python_script = "{0} {1} {2} {3} {4}".format(python_export_model,
                                                     _PARAM_INPUT_TYPE_IMAGE_TENSOR,
                                                     pipeline_parameter,
                                                     checkout_prefix,
                                                     output_directory)

        return "{0}; {1}".format(export_path, python_script)

    def export_model(self, model_number):
        file_name = "exported_model_{0}.{1}".format(model_number, "zip")
        archive_path = self.data_dir + "/data_dir"
        model_to_zip(
            file_name,
            archive_path,
            TF_API_CONFIG.format(self.data_dir),
            LABEL_FILE.format(self.data_dir),
            OUTPUT_DIRECTORY.format(self.data_dir, model_number, self._get_current_date())
        )

    # todo abdrashitov (Production preparation stage) extract to common utils
    @staticmethod
    def _get_current_date():
        now = datetime.datetime.now()
        return "{0}_{1}_{2}".format(now.day, now.month, now.year)


def find_train_iteration():
    iterations = []
    path = "{0}/data_dir/checkpoints/model.ckpt-*.index".format(os.environ.get("AIBUY_TENSORFLOW_DATA_DIR"))
    for name in glob.glob(path):
        m = re.search('-(\d+).', name)
        iterations.append(m.group(1))

    if len(iterations) == 0:
        print("There are no models for iteration finding in {0}".format(path))
        sys.exit()

    return max(iterations)


def check_env_variables():
    exit_flag = False
    if os.environ.get("AIBUY_TENSORFLOW_DATA_DIR") is None:
        print("Set environment variable AIBUY_TENSORFLOW_DATA_DIR")
        exit_flag = True
    elif os.environ.get("AIBUY_TENSORFLOW_ROOT_DIR") is None:
        print("Set environment variable AIBUY_TENSORFLOW_ROOT_DIR")
        exit_flag = True
    elif os.environ.get("PYTHONPATH") is None:
        print("Set environment variable PYTHONPATH")
        exit_flag = True

    if exit_flag:
        sys.exit()


if __name__ == '__main__':
    check_env_variables()
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', '-i',
                        help="the number of iteration",
                        required=False)
    args = parser.parse_args()

    iteration = args.iteration
    if args.iteration is None:
        iteration = find_train_iteration()

    print("Selected model iteration: {0}".format(iteration))
    GraphManagement().export(model_number=iteration)
