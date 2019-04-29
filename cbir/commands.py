import os
from pathlib import Path

import cbir
from cbir import DATABASES
from cbir.utils import basic
from cbir import cbir_core
from cbir.legacy_utils import draw_result

import cbir_evaluation.start_evaluation


def prepare_directory_structure(args):
    persistent_state = args.persistent_state or cbir.PERSISTENT_STATE
    databases = persistent_state / 'databases'
    if not os.path.exists(persistent_state):
        os.mkdir(persistent_state)
    if not os.path.exists(databases):
        os.mkdir(databases)


def _prepare_place_for_database(database):
    os.mkdir(Path(DATABASES) / database)


def _get_registered_databases():
    return [
        filename
        for filename in os.listdir(DATABASES)
        if os.path.isdir(Path(DATABASES) / filename)]


def _get_registered_cbir_indexes_of_database(database):
    return [
        filename
        for filename in os.listdir(Path(DATABASES) / database)
        if os.path.isdir(Path(DATABASES) / database / filename)]


####################


def build_cbir_index(database,
                     cbir_index_name,
                     path_to_images_to_index,
                     path_to_images_to_train_clusterer=None):
    """
    :param database:
    :param cbir_index_name
    :param path_to_images_to_index:
    :param path_to_images_to_train_clusterer:
    """
    path_to_images_to_train_clusterer = path_to_images_to_train_clusterer or path_to_images_to_index

    registered_databases = _get_registered_databases()
    if database not in registered_databases:
        _prepare_place_for_database(database)

    cbir_indexes_of_database = _get_registered_cbir_indexes_of_database(database)
    if cbir_index_name in cbir_indexes_of_database:
        message = f'Cbir index {cbir_index_name} for database {database} already exists.'
        raise ValueError(message)

    if not os.path.isdir(path_to_images_to_index):
        message = (f'Path to directory containing images must be provided.'
                   f'But {path_to_images_to_index} provided is not a directory')
        raise ValueError(message)

    os.mkdir(Path(DATABASES) / database / cbir_index_name)
    storage = cbir_core.Storage(Path(DATABASES) / database / cbir_index_name,
                                testing_path=path_to_images_to_index,
                                training_path=path_to_images_to_train_clusterer,
                                des_type='l2net',
                                label='',
                                max_keypoints=2000,
                                L=2,
                                K=10,
                                extensions=['jpg'],
                                debug=True)


def search(database,
           cbir_index_name,
           query,
           debug=False,
           **kwargs):
    """
    :param database:
    :param cbir_index_name:
    :param query:
    """
    registered_databases = _get_registered_databases()
    if database not in registered_databases:
        message = f'Database {database} has not been registered yet.'
        raise ValueError(message)

    cbir_indexes_of_database = _get_registered_cbir_indexes_of_database(database)
    if cbir_index_name not in cbir_indexes_of_database:
        message = f'Cbir index {cbir_index_name} for database {database} has not been created yet.'
        raise ValueError(message)

    # TODO: Make more robust loading of a storage object.
    # Now it seems like it can begin to build new Storage.
    storage = cbir_core.Storage(Path(DATABASES) / database / cbir_index_name,
                                testing_path=None,
                                training_path=None,
                                des_type='l2net',
                                label='',
                                max_keypoints=2000,
                                L=2,
                                K=10,
                                extensions=['jpg'],
                                debug=True)

    if not query:
        raise ValueError('No query provided')
    if not os.path.exists(query):
        message = f'File {query} does not exist'
        raise ValueError(message)
    if not basic.is_image(query):
        message = f'File {query} is not an image'
        raise ValueError(message)

    similar = storage.get_similar(query, topk=5,
                                  n_candidates=50, debug=debug,
                                  qe_enable=False)
    images = [v[1] for v in list(zip(*similar))[0]]

    if debug:
        print(f'Result images: {images}')
        draw_result(query, images)

    return images


def compute_descriptors(database,
                        cbir_index_name,
                        list_paths_to_images_to_compute_descriptors_for,
                        directory_or_list):

    options = ['directory', 'list']
    if directory_or_list not in options:
        raise ValueError(f'Provided {directory_or_list} not in list of possible options')
    raise NotImplemented


def build_index_and_inverted_index(database,
                                   cbir_index_name,
                                   path_to_images_to_index,
                                   path_to_images_to_train_clusterer=None):
    raise NotImplemented


def add_image(database,
              cbir_index_name,
              path_to_image_to_add):
    raise NotImplemented


def add_images(database,
               cbir_index_name,
               list_paths_to_images_to_add):
    raise NotImplemented



def evaluate(args):
    cbir_evaluation.start_evaluation.do_train_test(args.train_dir, args.test_dir, args.gt_dir,
                                                   args.sample)
