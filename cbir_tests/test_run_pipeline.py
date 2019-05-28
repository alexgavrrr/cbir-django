import logging
from pathlib import Path

import cbir
from cbir import BASE_DIR
from cbir.cbir_core import CBIRCore

logger = logging.getLogger()

CBIR_DATABASE_NAME = 'first'
CBIR_INDEX_NAME = 'index_first'
WHERE_PHOTOS = str(Path(BASE_DIR) / 'public' / 'media'
                   / 'content' / CBIR_DATABASE_NAME / 'database_all')
WHERE_PHOTOS_RELATIVE_TO_BASE_DIR = str(Path(WHERE_PHOTOS).relative_to(BASE_DIR))

from cbir.legacy_utils import find_image_files


def test_CBIR():
    database = CBIR_DATABASE_NAME
    name = CBIR_INDEX_NAME

    des_type = 'l2net'
    max_keypoints = 2000
    K = 10
    L = 4

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

    query = str(Path(WHERE_PHOTOS) / 'all_souls_000000.jpg')
    result_images = cbir_index.search(query)
    print(result_images)


if __name__ == '__main__':
    test_CBIR()
