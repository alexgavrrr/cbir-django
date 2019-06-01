import logging
from pathlib import Path

import cbir
from cbir import BASE_DIR
from cbir.cbir_core import CBIRCore
from cbir.cbir_core_external import CBIRCoreExt


logger = logging.getLogger()

CBIR_DATABASE_NAME = 'd-1'
CBIR_INDEX_NAME = 'i-exp'
WHERE_PHOTOS = str(Path(BASE_DIR) / 'public' / 'media'
                   / 'content' / CBIR_DATABASE_NAME / 'database_all')
WHERE_PHOTOS_RELATIVE_TO_BASE_DIR = str(Path(WHERE_PHOTOS).relative_to(BASE_DIR))

from cbir.legacy_utils import find_image_files


def test_CBIRCoreExt():
    database = CBIR_DATABASE_NAME
    name = CBIR_INDEX_NAME

    des_type = 'sift'
    max_keypoints = 2000
    K = 10
    L = 4

    list_paths_to_images_to_train_clusterer = find_image_files(WHERE_PHOTOS, cbir.IMAGE_EXTENSIONS, recursive=False)
    list_paths_to_images_to_index = list_paths_to_images_to_train_clusterer

    CBIRCoreExt.create_empty_if_needed(database, name,
                                    des_type=des_type, max_keypoints=max_keypoints,
                                    K=K, L=L)

    cbir_index = CBIRCoreExt.get_instance(database, name)
    cbir_index.compute_descriptors(list(set(list_paths_to_images_to_index)
                                        | set(list_paths_to_images_to_train_clusterer)),
                                   to_index=True,
                                   for_training_clusterer=True)
    cbir_index.train_clusterer()
    cbir_index.add_images_to_index()

    query = str(Path(WHERE_PHOTOS) / 'all_souls_000000.jpg')
    result_images = cbir_index.search(query)
    print(result_images)


def test_CBIRCore():
    database = CBIR_DATABASE_NAME
    name = CBIR_INDEX_NAME

    des_type = 'sift'
    max_keypoints = 4
    K = 10
    L = 1

    list_paths_to_images_to_train_clusterer = find_image_files(WHERE_PHOTOS, cbir.IMAGE_EXTENSIONS, recursive=False)
    list_paths_to_images_to_index = list_paths_to_images_to_train_clusterer

    CBIRCore.create_empty_if_needed(database, name,
                                    des_type=des_type, max_keypoints=max_keypoints,
                                    K=K, L=L)
    cbir_index = CBIRCore.get_instance(database, name)
    cbir_index.compute_descriptors(list(set(list_paths_to_images_to_index)
                                        | set(list_paths_to_images_to_train_clusterer)),
                                   to_index=True,
                                   for_training_clusterer=True)
    cbir_index.train_clusterer()
    cbir_index.add_images_to_index()

    show(database, name)

    # Add new images to index
    new_database_name = 'd-2'
    new_where_photos = str(Path(BASE_DIR) / 'public' / 'media'
                       / 'content' / new_database_name / 'database_all')

    list_paths_to_images_to_add_to_index = find_image_files(new_where_photos, cbir.IMAGE_EXTENSIONS, recursive=False)
    cbir_index.compute_descriptors(list_paths_to_images_to_add_to_index,
                                   to_index=True,
                                   for_training_clusterer=False)
    cbir_index.add_images_to_index()

    show(database, name)

    query = str(Path(WHERE_PHOTOS) / 'paris_defense_000216.jpg')
    result_images = cbir_index.search(
        query,
        n_candidates=3,
        topk=2,
        sv_enable=True,
        qe_enable=True)
    print(result_images)


def show(database, name):
    cbir_core = CBIRCore.get_instance(database, name)
    cbir_core.set_bow(cbir_core.load_bow())
    cbir_core.set_inv(cbir_core.load_inv())
    print(f'cbir_core.bow: {cbir_core.bow.toarray()}, {type(cbir_core.bow)}')
    print(f'cbir_core.inv: {cbir_core.inv}, {type(cbir_core.inv)}')
    params = cbir_core.load_params()
    data_dependent_params = cbir_core.load_data_dependent_params()
    print(f'params: {params}')
    print(f'data_dependent_params: {data_dependent_params}')

    photos = cbir_core.get_indexed_photos()
    print('\n'.join(map(str, photos)))



if __name__ == '__main__':
    import os
    os.chdir('..')
    print(os.getcwd())

    from cbir.configuration import configure_logging
    configure_logging('INFO')

    test_CBIRCore()
