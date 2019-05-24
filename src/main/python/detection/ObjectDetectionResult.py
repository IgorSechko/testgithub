class ObjectDetectionResult:
    def __init__(self, img_url, category, box) -> None:
        self.img_url = img_url
        self.category = category
        self.box = box
