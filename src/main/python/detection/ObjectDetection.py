from src.main.python.common.ImageUtils import get_images_by_boxes, image_by_urls
from src.main.python.detection.Box import Box
from src.main.python.detection.ObjectDetectionResult import ObjectDetectionResult


class ObjectDetection:
    def __init__(self) -> None:
        super().__init__()

    def prediction(self, img_url_list):
        img_list = image_by_urls(img_url_list)
        return self._prediction(img_list)

    def _prediction(self, img_list):
        # todo: Abdrashitov: method for object detection
        result = []
        for img_url, img in img_list:
            obj_detection = [ObjectDetectionResult(img_url=img_url, category="skirt", box=Box(158, 244, 200, 200))]
            result.append(get_images_by_boxes(img, obj_detection))
        return result
