import os
from pathlib import Path

import cbir
import cbir_evaluation.start_evaluation
from cbir import DATABASES


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
        train_dir = str(Path(data_building_root) / paris / 'jpg')
        test_dir = str(Path(data_building_root) / oxford / 'jpg')
        gt_dir = str(Path(data_building_root) / oxford / 'gt')

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
        **kwargs):
    train_dir, test_dir, gt_dir = _init_dirs(train_dir, test_dir, gt_dir, is_sample)
    cbir_evaluation.start_evaluation.start_train_test(
        train_dir,
        test_dir,
        gt_dir,
        des_type=des_type,
        sv=sv,
        qe=qe)
