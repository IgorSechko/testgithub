import os
import shutil


def clear_data_dir(path_to_folder):
    if not os.path.exists(path_to_folder):
        return

    shutil.rmtree(path_to_folder)
