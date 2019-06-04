import xmlrpc
import xmlrpc.client

import numpy as np
import os
from pathlib import Path
import logging

import cbir
import cbir_evaluation.start_evaluation
from cbir import DATABASES
from cbir.cbir_core import CBIRCore
from cbir.legacy_utils import find_image_files, find_image_files_bounded
from cbir.utils.basic import timeit_my


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

def prepare_cbir_directory_structure(persistent_state, **kwargs):
    persistent_state = persistent_state or cbir.PERSISTENT_STATE
    databases = persistent_state / 'databases'
    if not os.path.exists(persistent_state):
        os.mkdir(persistent_state)
    if not os.path.exists(databases):
        os.mkdir(databases)


def _init_dirs(train_dir, test_dir, gt_dir, is_sample):
    exists_none = False
    exists_not_none = False
    for now in [train_dir, test_dir, gt_dir]:
        exists_none |= now is None
        exists_not_none |= now is not None
    if exists_none and exists_not_none:
        raise ValueError

    if exists_none:
        suffix = "_sample" if is_sample else ""
        paris = 'Paris' + suffix
        oxford = 'Oxford' + suffix

        # data_building_root = str(Path(cbir.BASE_DIR) / 'data' / 'Buildings' / 'Original')
        data_building_root = str(Path('data') / 'Buildings' / 'Original')
        train_dir = str(Path(data_building_root) / oxford / 'jpg')
        test_dir = str(Path(data_building_root) / paris / 'jpg')
        gt_dir = str(Path(data_building_root) / paris / 'gt')

    return train_dir, test_dir, gt_dir


def evaluate_with_all_descriptors(
        train_dir,
        test_dir,
        gt_dir,
        is_sample,
        **kwargs):
    train_dir, test_dir, gt_dir = _init_dirs(train_dir, test_dir, gt_dir, is_sample)
    cbir_evaluation.start_evaluation.start_train_test_all_descriptors_and_modes(
        train_dir,
        test_dir,
        gt_dir)


def evaluate(
        train_dir,
        test_dir,
        gt_dir,
        is_sample,
        des_type,
        sv,
        qe,
        topk,
        **kwargs):
    train_dir, test_dir, gt_dir = _init_dirs(train_dir, test_dir, gt_dir, is_sample)
    cbir_evaluation.start_evaluation.start_train_test(
        train_dir,
        test_dir,
        gt_dir,
        des_type=des_type,
        sv=sv,
        qe=qe,
        topk=topk)


def evaluate_only(
        database_name, index_name, database_photos_dir, gt_dir,
        sv, qe,
        p_fine_max,
        topk,
        **kwargs):
    cbir_evaluation.start_evaluation.start_test(
        database_name, index_name, database_photos_dir, gt_dir,
        sv, qe,
        p_fine_max=p_fine_max,
        topk=topk)


def create_empty_if_needed(
        database_name, index_name,
        des_type, max_keypoints,
        K, L,
        **kwargs):
    des_type = des_type or cbir.DES_TYPE
    max_keypoints = max_keypoints or cbir.MAX_KEYPOINTS
    CBIRCore.create_empty_if_needed(database_name, index_name,
                                    des_type=des_type,
                                    max_keypoints=max_keypoints,
                                    K=K, L=L)


def compute_descriptors(
        database_name, index_name,
        path_to_1_images,
        path_to_distractor_images,
        path_to_2_images,
        max_images,
        **kwargs):

    cbir_core = CBIRCore.get_instance(database_name, index_name)

    list_paths_to_1_images = find_image_files_bounded(
        path_to_1_images,
        cbir.IMAGE_EXTENSIONS,
        recursive=False,
        max_images=max_images)

    max_images_2 = None
    if max_images is not None:
        max_images_2 = max_images - len(list_paths_to_1_images)
    list_paths_to_2_images = find_image_files_bounded(
        path_to_2_images,
        cbir.IMAGE_EXTENSIONS,
        recursive=False,
        max_images=max_images_2)

    distractor_max_images = None
    if max_images is not None:
        distractor_max_images = max_images - (len(list_paths_to_1_images) + len(list_paths_to_2_images))
    list_paths_to_distractor_images = find_image_files_bounded(
        path_to_distractor_images,
        cbir.IMAGE_EXTENSIONS,
        recursive=True,
        max_images=distractor_max_images)

    cbir_core.compute_descriptors(list_paths_to_1_images,
                                  to_index=True,
                                  for_training_clusterer=True)

    cbir_core.compute_descriptors(list_paths_to_2_images,
                                  to_index=True,
                                  for_training_clusterer=False)

    cbir_core.compute_descriptors(list_paths_to_distractor_images,
                                  to_index=True,
                                  for_training_clusterer=True)


def train_clusterer(
        database_name, index_name,
        **kwargs):
    cbir_core = CBIRCore.get_instance(database_name, index_name)
    cbir_core.train_clusterer()


def compute_bow_and_inv(
        database_name, index_name,
        **kwargs):
    cbir_core = CBIRCore.get_instance(database_name, index_name)
    cbir_core.add_images_to_index()


def create_index(
        database_name, index_name,
        path_to_1_images,
        path_to_distractor_images,
        path_to_2_images,
        max_images,
        des_type, max_keypoints,
        K, L,
        only_retrain_vocab,
        only_bow_inv,
        **kwargs):

    if not (only_retrain_vocab or only_bow_inv):
        create_empty_if_needed(
            database_name, index_name,
            des_type, max_keypoints,
            K, L, )
        elapsed, _ = timeit_my(compute_descriptors)(
            database_name, index_name,
            path_to_1_images,
            path_to_distractor_images,
            path_to_2_images,
            max_images)
        logging.getLogger('profile.computing_descriptors').info(f'{elapsed}')
        train_clusterer(database_name, index_name)
        compute_bow_and_inv(database_name, index_name)
    elif only_retrain_vocab:
        # if only_retrain_vocab then set new K, L
        # (ignore des_type and max_keypoints)
        cbir_core = CBIRCore.get_instance(database_name, index_name)
        if cbir_core.K != K or cbir_core.L != L:
            cbir_core.set_K_L(K, L)
        train_clusterer(database_name, index_name)
        compute_bow_and_inv(database_name, index_name)
    else:
        # only_bow_inv
        cbir_core = CBIRCore.get_instance(database_name, index_name)
        cbir_core.reset_bow_inv()
        compute_bow_and_inv(database_name, index_name)


def change_params(
        database_name, index_name,
        des_type, max_keypoints,
        K, L,
        **kwargs):
    if not CBIRCore.exists(database_name, index_name):
        raise ValueError(f'{database_name} {index_name} does not exists. You can just create index')

    logger = logging.getLogger('change_params')

    cbir_core = CBIRCore.get_instance(database_name, index_name)
    params = cbir_core.load_params()
    data_dependent_params = cbir_core.load_params()

    if des_type:
        if des_type != params['des_type']:
            raise ValueError('Another des_type')
        params['des_type'] = des_type

    if max_keypoints:
        if max_keypoints != params['max_keypoints']:
            raise ValueError('Another max_keypoints')
        params['max_keypoints'] = max_keypoints

    logger.info('Setting new params')
    params['K'] = K
    params['L'] = L
    params['n_words'] = K ** L

    data_dependent_params['idf'] = np.zeros(params['n_words'], dtype=np.float32)
    data_dependent_params['freqs'] = np.zeros(params['n_words'], dtype=np.int32)
    data_dependent_params['most_frequent'] = []
    data_dependent_params['least_frequent'] = []

    CBIRCore._save_params(database_name, index_name, params)
    CBIRCore._save_data_dependent_params(database_name, index_name, data_dependent_params)
    clusterer = None
    CBIRCore._save_clusterer(database_name, index_name, clusterer)
    cbir_core = CBIRCore.get_instance(database_name, index_name)

    cbir_core.clean_bow_and_inv()
    cbir_core.train_clusterer()
    cbir_core.add_images_to_index()


def search(
        database_name, index_name, query,
        n_candidates, topk,
        sv, qe,
        p_fine_max,
        cbir_server_port=None,
        **kwargs):
    cbir_server_port = cbir_server_port or 8701

    search_params = {}
    if n_candidates:
        search_params['n_candidates'] = n_candidates
    if topk:
        search_params['topk'] = topk
    search_params['sv_enable'] = sv
    search_params['qe_enable'] = qe
    search_params['p_fine_max'] = p_fine_max

    cbir_server = xmlrpc.client.ServerProxy(f'http://localhost:{cbir_server_port}', allow_none=True)
    result = cbir_server.search(database_name, index_name, query, search_params)
    print(result)


def run_server(
        port,
        nthreads,
        **kwargs, ):
    import cbir.server
    cbir.server.run(port, nthreads)


def run_client(
        port,
        database,
        index,
        query,
        topk,
        sv, qe,
        **kwargs, ):
    import cbir.client
    cbir.client.run(
        port,
        database, index,
        query,
        topk,
        sv, qe)
