import glob
import os
import shutil
import zipfile


def clear_data_dir(path_to_folder):
    if not os.path.exists(path_to_folder):
        return

    shutil.rmtree(path_to_folder)


def model_to_zip(archive_name,
                 archive_path,
                 tf_api_config,
                 label,
                 graph):
    path = '{0}/{1}'.format(archive_path, archive_name)
    archive = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)

    _add_file_to_zip(archive, tf_api_config)
    _add_file_to_zip(archive, label)

    _add_directory_to_zip(archive, graph)
    archive.close()


def _add_to_zip(archive, path):
    for file in _files_in_directory(_add_prefix(path)):
        if os.path.isdir(file):
            _add_to_zip(archive, _add_prefix(file))
        archive.write(file, _remove_full_path(file))


def _add_file_to_zip(archive, file):
    archive.write(file, _remove_full_path(file))


# todo Production preparation stage
def _add_directory_to_zip(archive, directory):
    for file in _files_in_directory(_add_prefix(directory)):
        if os.path.isdir(file):
            for inner_file in _files_in_directory(_add_prefix(file)):
                archive.write(inner_file, _remove_full_path(file) + "/" + _remove_full_path(inner_file))
        archive.write(file, _remove_full_path(file))


def _remove_full_path(file_name):
    return file_name.split(os.sep)[-1]


def _add_prefix(path):
    return path + "/*"


def _files_in_directory(path):
    return glob.glob(path)
