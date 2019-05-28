import argparse
from os.path import expanduser

from src.main.python.service.ObjectDetectionService import ObjectDetectionService

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', '-u',
                        type=list,
                        help="urls",
                        required=False)
    args = parser.parse_args()
    object_detection_service = ObjectDetectionService(
        "{0}{1}/content/inference_graph/graph3128_28_5_2019/frozen_inference_graph.pb".format(expanduser("~"), "/git"),
        "{0}{1}/content/data_dir/new/label_map.pbtxt".format(expanduser("~"), "/git"))
    list = []
    list.append('https://lsco.scene7.com/is/image/lsco/Levis/clothing/228500035-front-pdp.jpg')
    detections = object_detection_service.prediction(list)
    for detection in detections:
        print(detection[1])
