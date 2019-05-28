import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from src.main.python.common.tensorflow import label_map_util, string_int_label_map_pb2
from src.main.python.common.tensorflow.label_map_util import validate_label_map


class ObjectDetection:

    def __init__(self, model_graph, label_path):
        self.PATH_TO_CKPT = model_graph
        self.PATH_TO_LABELS = label_path
        self.category_index = None
        self.img_detections = None
        self.init_category()

    def init_category(self):
        print("Init category...")
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.category_index = [(item.id, item.name) for item in
                               label_map.item]

    def load_labelmap(self, path):
        """Loads label map proto.

        Args:
          path: path to StringIntLabelMap proto text file.
        Returns:
          a StringIntLabelMapProto
        """
        with tf.gfile.GFile(path, 'r') as fid:
            label_map_string = fid.read()
            label_map = string_int_label_map_pb2.StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)
        validate_label_map(label_map)
        return label_map

    def _get_img_array(self, image):
        (img_width, img_height) = image.size
        image_np = np.array(image.getdata()).reshape((img_width, img_height, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        return image_np_expanded

    def object_detection(self, image):
        """Запускаем детектор картинок"""
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        img_detections = []

        prepared_img_array = [{
            'img_array': self._get_img_array(image)
        }]
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                for image in prepared_img_array:
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image['img_array']})
                    for index, value in enumerate(classes[0]):
                        try:
                            item = self.category_index[int(value) - 1]
                            img_detections.append({
                                'category_name': item[1],
                                'category_id': item[0],
                                'category_proba': scores[0, index],
                                'category_box': boxes[0, index],
                                'img_array': image['img_array']
                            })
                        except:
                            print("Wrong label file")
        # return result
        self.img_detections = img_detections
        return img_detections
