import datetime
import os

from TFFashionDetection.utils.CommonUtils import load_config


class GraphManagement:

    def __init__(self) -> None:
        config = load_config()
        self.root_dir = config['root_dir']

    def export(self, model_number):
        # Create graph from checkpoints
        try:
            os.system(self._create_export_script(model_number))
            print("Model was exported")
        except Exception as e:
            print("Process error")

    def _create_export_script(self, model_number):
        export_path = "export PYTHONPATH=$PYTHONPATH:{0}/models/research/slim:{0}/models/research/".format(
            self.root_dir)

        python_export_model = "python {0}/models/research/object_detection/export_inference_graph.py".format(
            self.root_dir)
        input_parameter = "--input_type image_tensor"
        pipeline_parameter = "--pipeline_config_path={0}/data_dir/tf_api.config".format(self.root_dir)
        checkout_prefix = "--trained_checkpoint_prefix={0}/data_dir/new/checkpoints/model.ckpt-{1}".format(
            self.root_dir,
            model_number)
        output_directory = "--output_directory {0}/inference_graph/graph{1}_{2}".format(self.root_dir, model_number,
                                                                                        self._get_current_date())

        python_script = "{0} {1} {2} {3} {4}".format(python_export_model, input_parameter, pipeline_parameter,
                                                     checkout_prefix,
                                                     output_directory)
        return "{0}; {1}".format(export_path, python_script)

    # todo abdrashitov (Production preparation stage) extract to common utils
    @staticmethod
    def _get_current_date():
        now = datetime.datetime.now()
        return "{0}_{1}_{2}".format(now.day, now.month, now.year)


if __name__ == '__main__':
    GraphManagement().export(model_number="11241")
