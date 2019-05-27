"""
    Object detection from frozen graph
"""

from src.main.python.common.ImageUtils import get_images_by_boxes, image_by_urls
from src.main.python.detection.Box import Box
from src.main.python.detection.ObjectDetection import ObjectDetection
from src.main.python.detection.ObjectDetectionResult import ObjectDetectionResult


class ObjectDetectionService:

    def __init__(self, model_graph, label_path) -> None:
        self.object_detector = ObjectDetection(
            model_graph=model_graph,
            label_path=label_path
        )

    def prediction(self, img_url_list):
        img_list = image_by_urls(img_url_list)
        return self._prediction(img_list)

    def _prediction(self, url_list):
        result = []
        for url, img in url_list:
            predictions = self.object_detector.object_detection(img)[:2]
            obj_detections = self._prepare_result(predictions, url)
            result.append((url, get_images_by_boxes(img, obj_detections)))
        return result

    def _prepare_result(self, predictions, img_url):
        result = []
        for prediction in predictions:
            x, y, width, height = self._prepare_box(prediction['category_box'], prediction['img_array'])
            result.append(ObjectDetectionResult(img_url=img_url, category=prediction['category_name'],
                                                box=Box(x, y, width, height)))
        return result

    def _prepare_box(self, category_box, image_array):
        # relative to absolute coords
        im_width, im_height = image_array[0].shape[:2]
        ymin, xmin, ymax, xmax = category_box
        return int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)


# if __name__ == '__main__':
#     root_dir = os.environ.get("AIBUY_TENSORFLOW_ROOT_DIR")
#     model_graph = '{0}/output_inference_graph.pb'.format(root_dir)
#     label_path = '{0}/data_dir/annotations/label_map.pbtxt'.format(root_dir)
#     detector = ObjectDetectionService(model_graph, label_path)
#     detector.object_detection(image_dir='{0}/data_dir/images'.format(root_dir))
