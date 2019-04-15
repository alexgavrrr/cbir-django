import os
from pathlib import Path
import shutil

import cbir
from cbir import DATABASES
from cbir import QUERIES
from cbir.utils import basic
from cbir import photo_storage_inverted_file
from cbir.legacy_utils import draw_result
from cbir.session import Session
from cbir import exceptions

import cbir_tests.start_evaluation


def _get_registered_databases():
    return [
        filename
        for filename in os.listdir(DATABASES)
        if os.path.isdir(Path(DATABASES) / filename)]


def register(args):
    """

    :param args.database:
    :param args.path:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """
    registered_databases = _get_registered_databases()
    database = args.database or cbir.CONFIG['database']
    if database in registered_databases:
        message = f'Databese {database} already exists.'
        raise ValueError(message)

    path_to_images = args.path
    if not os.path.isdir(path_to_images):
        message = f'Path to directory containing images must be provided.'
        raise ValueError(message)

    os.mkdir(Path(DATABASES) / database)
    basic.copy_files(path_to_images,
                     Path(DATABASES) / database,
                     predicate=basic.is_image,
                     prefix_for_conflicted_files=None)

    # TODO: Run indexing of images.
    storage = photo_storage_inverted_file.Storage(Path(DATABASES) / database,
                                                  Path(DATABASES) / database,
                                                  'l2net',
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


def reindex_database(args):
    """

    :param args.database:
    :param args.path:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """


def search(args, debug=True):
    """
    :param args.database:
    :param args.query:
    :param args.save:
    :param args.tag:

    If there are no required parameters in args then they may be in cbir.CONFIG
    """
    database = args.database or cbir.CONFIG['database']
    registered_databases = _get_registered_databases()
    if database not in registered_databases:
        message = f'Databese {database} does not exist.'
        raise ValueError(message)

    # TODO: Make more robust loading of a storage object.
    # Now it seems like it can begin to build new Storage.
    storage = photo_storage_inverted_file.Storage(Path(DATABASES) / database,
                                                  Path(DATABASES) / database,
                                                  'l2net',
                                                  label='',
                                                  max_keypoints=2000,
                                                  L=2,
                                                  K=10,
                                                  extensions=['jpg'],
                                                  debug=True)

    if not args.query:
        args.query = input('Please enter path to query image...\n')
    if not os.path.exists(args.query):
        message = f'File {args.query} does not exist'
        raise ValueError(message)
    if not basic.is_image(args.query):
        message = f'File {args.query} is not an image'
        raise ValueError(message)

    if args.save:
        query_filename_text, extension = os.path.splitext(Path(args.query).name)
        destination = Path(QUERIES) / query_filename_text

        if os.path.exists(f'{destination}{extension}'):
            destination_original = destination
            attempt = 1
            destination = f'{destination_original}_{attempt}'
            while os.path.exists(f'{destination}{extension}'):
                attempt += 1
                destination = f'{destination_original}_{attempt}'

        shutil.copyfile(args.query, f'{destination}{extension}')
        args.query = f'{destination}{extension}'

    session = Session(database, args.query, query_saved=args.save)
    similar = storage.get_similar(args.query, topk=5,
                                  n_candidates=50, debug=debug,
                                  qe_enable=False)
    images = [v[1] for v in list(zip(*similar))[0]]

    if debug:
        print(f'Result images: {images}')
        draw_result(args.query, images)

    session.add_list(images)

    if args.save:
        args.tag = args.tag or input('Please enter tag for session...\n')
        while True:
            try:
                session.save(args.tag)
            except exceptions.DuplicateSessionTag:
                args.tag = input(f'Tag {args.tag} already exists. Please enter new...\n')
            else:
                break

    return images


def show(args):
    """
    :param args.database:
    :param args.tag:
    """
    database = args.database or cbir.CONFIG['database']
    registered_databases = _get_registered_databases()
    if database not in registered_databases:
        message = f'Databese {database} does not exist.'
        raise ValueError(message)

    session = Session.load(args.database, args.tag)
    print(session.basket)
    draw_result(session.query, session.basket)


def evaluate(args):
    cbir_tests.start_evaluation.do_train_test(args.train_dir, args.test_dir, args.gt_dir,
                                              args.sample)
