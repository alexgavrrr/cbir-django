import os
from pathlib import Path

import cbir
from cbir import DATABASES
from cbir.utils import basic
from cbir import photo_storage_inverted_file
from cbir.legacy_utils import draw_result

import cbir_evaluation.start_evaluation


def _get_registered_databases():
    return [
        filename
        for filename in os.listdir(DATABASES)
        if os.path.isdir(Path(DATABASES) / filename)]


def _prepare_place_for_database(database):
    os.mkdir(Path(DATABASES) / database)


def _get_registered_cbir_indexes_of_database(database):
    return [
        filename
        for filename in os.listdir(Path(DATABASES) / database)
        if os.path.isdir(Path(DATABASES) / database / filename)]


def register(args):
    """

    :param args.database:
    :param args.cbir_index_name
    :param args.path:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """
    registered_databases = _get_registered_databases()
    database = args.database
    if database not in registered_databases:
        _prepare_place_for_database(database)

    cbir_index_name = args.cbir_index_name
    cbir_indexes_of_database = _get_registered_cbir_indexes_of_database(database)
    if cbir_index_name in cbir_indexes_of_database:
        message = f'Cbir index {cbir_index_name} for database {database} already exists.'
        raise ValueError(message)

    path_to_images = args.path
    if not os.path.isdir(path_to_images):
        message = (f'Path to directory containing images must be provided.'
                   f'But {path_to_images} provided is not a directory')
        raise ValueError(message)

    os.mkdir(Path(DATABASES) / database / cbir_index_name)
    storage = photo_storage_inverted_file.Storage(Path(DATABASES) / database / cbir_index_name,
                                                  testing_path=path_to_images,
                                                  training_path=path_to_images,
                                                  des_type='l2net',
                                                  label='',
                                                  max_keypoints=2000,
                                                  L=2,
                                                  K=10,
                                                  extensions=['jpg'],
                                                  debug=True)


def add_images(args):
    """

    :param args.database:
    :param args.path:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """


def search(args, debug=False):
    """
    :param args.database:
    :param args.cbir_index_name:
    :param args.path:
    :param args.query:
    :param args.save:
    :param args.tag:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """
    database = args.database
    registered_databases = _get_registered_databases()
    if database not in registered_databases:
        message = f'Database {database} has not been registered yet.'
        raise ValueError(message)

    cbir_index_name = args.cbir_index_name
    cbir_indexes_of_database = _get_registered_cbir_indexes_of_database(database)
    if cbir_index_name not in cbir_indexes_of_database:
        message = f'Cbir index {cbir_index_name} for database {database} has not been created yet.'
        raise ValueError(message)

    path_to_images = args.path
    if not os.path.isdir(path_to_images):
        message = (f'Path to directory containing images must be provided.'
                   f'But {path_to_images} provided is not a directory')
        raise ValueError(message)

    # TODO: Make more robust loading of a storage object.
    # Now it seems like it can begin to build new Storage.
    storage = photo_storage_inverted_file.Storage(Path(DATABASES) / database / cbir_index_name,
                                                  testing_path=path_to_images,
                                                  training_path=None,
                                                  des_type='l2net',
                                                  label='',
                                                  max_keypoints=2000,
                                                  L=2,
                                                  K=10,
                                                  extensions=['jpg'],
                                                  debug=True)

    if not args.query:
        raise ValueError('No query provided')
    if not os.path.exists(args.query):
        message = f'File {args.query} does not exist'
        raise ValueError(message)
    if not basic.is_image(args.query):
        message = f'File {args.query} is not an image'
        raise ValueError(message)

    similar = storage.get_similar(args.query, topk=5,
                                  n_candidates=50, debug=debug,
                                  qe_enable=False)
    images = [v[1] for v in list(zip(*similar))[0]]

    if debug:
        print(f'Result images: {images}')
        draw_result(args.query, images)

    return images


def evaluate(args):
    cbir_evaluation.start_evaluation.do_train_test(args.train_dir, args.test_dir, args.gt_dir,
                                                   args.sample)


def prepare_directory_structure(args):
    persistent_state = args.persistent_state or cbir.PERSISTENT_STATE
    databases = persistent_state / 'databases'
    if not os.path.exists(persistent_state):
        os.mkdir(persistent_state)
    if not os.path.exists(databases):
        os.mkdir(databases)
