import json
import os
from os.path import expanduser


def load_config():
    """Загружаем JSON-конфиг со служебной информацией"""
    # TODO: определять внутри скрипта, откуда он запускается
    return json.loads(
        open(os.path.join(f'{expanduser("~")}/app/content', 'TFFashionDetection', 'etc/directory_conf.json'),
             'r').read())
