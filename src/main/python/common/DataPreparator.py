import io
import logging
import os
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
from PIL import Image as Pil_image
from lxml import etree
from tqdm import tqdm

from src.main.python.common.tensorflow import dataset_util

API_PATH = os.path.join('/content', 'models/research')
sys.path.append(API_PATH)

logger = logging.getLogger(__name__)


class DataPreparator:
    """Класс для подготовки DeepFashion датасета"""

    def __init__(self, data_dir, metadata_dir, allowed_categories):
        self.metadata_dir = metadata_dir
        self.fashion_data = os.path.join(data_dir, 'fashion_data')

        self.clothes_to_category = None
        self.img_to_category = None
        self.img_to_eval = None
        self.img_index = None
        self.bboxes = None

        # категории для обучения
        self.allowed_categories = allowed_categories
        self.label_mapping = None

        # вспомогательные функции для чтения файлов DeepFashion
        self.row_processors = {
            'list_bbox': lambda row: {row[0]: [int(i) for i in row[1:]]},
            'list_category_cloth': lambda row: {row[0].lower(): int(row[1])},
            'list_category_img': lambda row: {row[0].lower(): int(row[1])},
            'list_eval_partition': lambda row: {row[0]: row[1]}
        }

    def build(self):
        print("Save information about categories")
        self.create_tf_dirs()
        self.deep_fashion_data_structure()
        self.prepare_img_index()
        self.create_tf_records(['train', 'test'])  # ещё есть val

    def create_tf_dirs(self):
        os.mkdir(self.metadata_dir)
        os.mkdir(os.path.join(self.metadata_dir, 'images'))
        os.mkdir(os.path.join(self.metadata_dir, 'annotations'))
        os.mkdir(os.path.join(self.metadata_dir, 'data'))
        os.mkdir(os.path.join(self.metadata_dir, 'checkpoints'))
        os.mkdir(os.path.join(self.metadata_dir, 'annotations', 'xmls'))
        print("Create metadata directory %s" % self.metadata_dir)

    def deep_fashion_data_structure(self):
        print("Reading Anno directory...")
        print("Reading list_box.txt...")
        bbbox_file = os.path.join(self.fashion_data, "Anno/list_bbox.txt")
        self.bboxes = self.read_configuration_file(bbbox_file, self.row_processors['list_bbox'])

        print("Reading list_category_cloth.txt...")
        clothes_to_category_file = os.path.join(self.fashion_data, "Anno/list_category_cloth.txt")
        self.clothes_to_category = self.read_configuration_file(clothes_to_category_file,
                                                                self.row_processors['list_category_cloth'])
        self.clothes_to_category = {key: value for (key, value) in self.clothes_to_category.items() if
                                    key in self.allowed_categories}

        print("Reading list_category_img...")
        img_to_category_file = os.path.join(self.fashion_data, "Anno/list_category_img.txt")
        self.img_to_category = self.read_configuration_file(img_to_category_file,
                                                            self.row_processors['list_category_img'])

        print("Reading list_eval_partion...")
        img_to_eval_file = os.path.join(self.fashion_data, "Eval/list_eval_partition.txt")
        self.img_to_eval = self.read_configuration_file(img_to_eval_file, self.row_processors['list_eval_partition'])

        print("Prepare label mapping ...")
        self.label_mapping = {v: k for k, v in self.clothes_to_category.items()}

    def read_configuration_file(self, fname, row_processor=None):
        with open(fname) as f:
            _ = f.readline().strip()
            _ = f.readline().strip().split()
            lines = {}
            for line in f:
                row = line.strip().split()
                if row_processor is not None:
                    row = row_processor(row)
                lines.update(row)
            return lines

    def prepare_img_index(self):
        def search_category(f_name):
            corrected_name = "img/" + f_name.replace("_img_", "/img_")
            category = self.img_to_category[corrected_name]
            return category

        self.img_index = {
            k: {
                'eval': self.img_to_eval[k],
                'filename': ('_'.join(k.split('/')[-2:])).lower()
            }
            for k in tqdm(self.img_to_eval, total=len(self.img_to_eval), desc="Create Image index...")
        }

        # определяем класс изображения (по имени файла)
        [self.img_index[k].update({
            'class': search_category(self.img_index[k]['filename'])
        }) for k in tqdm(self.img_index, total=len(self.img_index), desc="Detect image class...")]

        self.img_index = {k: v for (k, v) in
                          tqdm(self.img_index.items(), total=len(self.img_index), desc="Clear smallest categories...")
                          if v['class'] in self.clothes_to_category.values()}

        self.balance_img_index()

        print(
            'Category distribution: %s' %
            Counter([j['class'] for i, j in self.img_index.items()])
        )

        print(
            'QA distribution: %s' %
            Counter([j['eval'] for i, j in self.img_index.items()])
        )

    def balance_img_index(self):
        evals = np.unique([j['eval'] for i, j in self.img_index.items()])
        classes = np.unique([j['class'] for i, j in self.img_index.items()])
        balansed_img_index = dict()
        for _eval in tqdm(evals, total=len(evals), desc="Balance dataset..."):
            # выборка с элементами выборки
            eval_examples = {i: j for i, j in self.img_index.items() if j['eval'] == str(_eval)}
            # статистика распределения по классам
            class_counter = Counter([j['class'] for i, j in eval_examples.items()])
            # находим класс, у которого меньше всего представителей
            min_elem_class, min_elem_num = list(class_counter.items())[0]
            for i, j in class_counter.items():
                min_elem_class, min_elem_num = (i, j) if j < min_elem_num else (min_elem_class, min_elem_num)
            # формируем новый словарь: выпиливаем "лишние" обучающие примеры чтобы классы были сбалансированы
            result_dict = dict()
            for _class in classes:
                class_examples = {i: j for i, j in eval_examples.items() if j['class'] == _class}
                if _class != min_elem_class:
                    # делаем downsample
                    class_subsample = np.random.choice(list(class_examples.keys()), size=min_elem_num, replace=False)
                    class_examples = {i: j for i, j in class_examples.items() if i in class_subsample}
                result_dict.update(class_examples)
            balansed_img_index.update(result_dict)
        self.img_index = balansed_img_index

    def get_img_descriptions(self, f_name, file_descr):
        """Получаем описания обучающего примера: в виде XML и TFRecord"""
        bbox = self.bboxes[f_name]

        filename = os.path.join(self.fashion_data, 'Img', f_name)
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Pil_image.open(encoded_jpg_io)

        img_shape = image.size + tuple([3])

        tf_record = self.create_tf_example(file_descr, bbox, img_shape, encoded_jpg)
        xml_record = self.create_xml_example(file_descr, bbox, img_shape)

        return tf_record, xml_record

    def create_tf_example(self, file_descr, bbox, img_shape, encoded_jpg):
        """создаём TFRecord для Tensorflow"""
        image_format = b'jpg'
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

        # на случай нескодьких детекций в одном файле можно тут добавить цикл, но у нас всего одна детекция в каждом файле
        ymin, xmin, ymax, xmax = bbox
        width, height, depth = img_shape

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(self.label_mapping[file_descr['class']].encode('utf8'))
        classes.append(file_descr['class'])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_descr['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(file_descr['filename'].encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example

    def create_xml_example(self, file_descr, bbox, img_shape):
        # создаём XML
        ymin, xmin, ymax, xmax = bbox
        width, height, depth = img_shape
        # create XML
        root_xml = etree.Element('annotation')
        # filename
        child = etree.Element('filename')
        child.text = file_descr['filename']
        root_xml.append(child)
        # size
        child = etree.Element('size')
        _width = etree.Element('width')
        _width.text = str(width)
        child.append(_width)
        _height = etree.Element('height')
        _height.text = str(height)
        child.append(_height)
        depth = etree.Element('depth')
        depth.text = '3'
        child.append(depth)
        root_xml.append(child)
        # segmented
        child = etree.Element('segmented')
        child.text = '0'
        root_xml.append(child)
        # object
        child = etree.Element('object')
        # name
        name = etree.Element('name')
        name.text = self.label_mapping[file_descr['class']]
        child.append(name)
        # bndbox -> ymin, xmin, ymax, xmax
        bndbox = etree.Element('bndbox')
        #
        _ymin = etree.Element('ymin')
        _ymin.text = str(ymin)
        bndbox.append(_ymin)
        #
        _xmin = etree.Element('xmin')
        _xmin.text = str(xmin)
        bndbox.append(_xmin)
        #
        _ymax = etree.Element('ymax')
        _ymax.text = str(ymax)
        bndbox.append(_ymax)
        #
        _xmax = etree.Element('xmax')
        _xmax.text = str(ymin)
        bndbox.append(_xmax)
        #
        child.append(bndbox)
        root_xml.append(child)

        # pretty string
        xml_str = etree.tostring(root_xml, pretty_print=True)
        return xml_str

    def create_tf_records(self, scenarios):
        """Создаём XML описаниями"""
        base_xml_path = os.path.join(self.metadata_dir, 'annotations', 'xmls')
        trainval_path = os.path.join(self.metadata_dir, 'annotations', 'trainval.txt')
        # trainval должны быть все названия, его не перезаписываем
        trainval_file = open(trainval_path, 'a')
        for scenario in tqdm(scenarios, total=len(scenarios), desc="Create XML scenarios"):
            print("Generate scenario description %s" % scenario)
            self.generate_files_by_scenario(scenario, trainval_file, base_xml_path)
        trainval_file.close()
        # записываем файл с метками классов
        label_map_path = os.path.join(self.metadata_dir, 'annotations', 'label_map.pbtxt')
        label_map_file = open(label_map_path, 'w')
        for k, v in self.label_mapping.items():
            label_map_file.write("""item { id: %s name: '%s'}\n""" % (k, v))
        label_map_file.close()
        print('XML was created in: %s' % base_xml_path)
        print('Classes mark file: %s' % label_map_path)

    def generate_files_by_scenario(self, scenario, trainval_descriptor, base_xml_path):
        """Генерим наборы файлов: XML+TFR"""
        tfr_out_path = os.path.join(self.metadata_dir, 'annotations', scenario + '.record')
        writer = tf.python_io.TFRecordWriter(tfr_out_path)
        img_keys = list(self.img_index.keys())
        np.random.shuffle(img_keys)
        for img_path in tqdm(img_keys, total=len(img_keys), desc="Generate dataset files: xml_tfr"):
            img_descr = self.img_index[img_path]
            if img_descr['eval'] == scenario:
                tf_example, xml_example = self.get_img_descriptions(img_path, img_descr)
                with open(os.path.join(base_xml_path, img_descr['filename'][:-4] + '.xml'), 'w') as xml_file:
                    xml_file.write(xml_example.decode("utf-8"))
                writer.write(tf_example.SerializeToString())
                trainval_descriptor.write(img_descr['filename'][:-4] + '\n')
                # # копируем изображение в директорию (пригодится для инференса)
                # shutil.copy(
                #     os.path.join(self.fashion_data, 'Img', img_path),
                #     os.path.join(self.metadata_dir, 'images', img_descr['filename'])
                # )
        writer.close()
        print('TFRecords was created: %s' % tfr_out_path)

    def get_category_count(self):
        return len(self.allowed_categories)
