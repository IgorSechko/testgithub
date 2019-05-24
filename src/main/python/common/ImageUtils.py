import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from PIL import Image as pil_image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array

_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}


def image_by_url(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return image_url, img
    except:
        print("Error -> {0}".format(image_url))


def image_by_urls(image_url_list):
    result = []
    for image_url in image_url_list:
        result.append(image_by_url(image_url))
    return result


def get_images_by_boxes(img, prediction_results):
    result = []
    for prediction in prediction_results:
        box_img = img.crop(box_to_area(prediction.box))
        box_img.show()
        result.append(box_img)
    return result


def box_to_area(box):
    return box.x, box.y, box.x + box.width, box.y + box.height


def to_target_size(target_size, img, interpolation='nearest'):
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        if interpolation not in _PIL_INTERPOLATION_METHODS:
            raise ValueError(
                'Invalid interpolation method {} specified. Supported '
                'methods are {}'.format(
                    interpolation,
                    ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
        resample = _PIL_INTERPOLATION_METHODS[interpolation]
        img = img.resize(width_height_tuple, resample)
    return np.array(img)


def prepare_image(img):
    img = to_target_size((224, 224), img)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_image_by_url(url):
    try:
        response = requests.get(url.images)
        img = pil_image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except:
        print("error converting img ${0}".format(url))
        return None


def get_image_bytes(image):
    try:
        response = requests.get(image)
        img = pil_image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except:
        print("error converting img ${0}".format(image))
        return None


def create_directory(file_path):
    try:
        os.makedirs(file_path)
    except OSError as e:
        pass
