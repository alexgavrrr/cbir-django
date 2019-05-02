import os
from pathlib import Path

import cbir
from cbir_evaluation.evaluation import (evaluate, evaluate_only)

MAX_KEYPOINTS = 2000
K = 10
L = 4


def do_train_test(train_dir, test_dir, gt_dir,
                  algo_params,
                  sv=True, qe=True):
    mAP, mAP_new = evaluate(train_dir, test_dir, gt_dir,
                    algo_params=algo_params,
                    sv_enable=sv, qe_enable=qe)
    return mAP, mAP_new


def start_train_test(
        train_dir,
        test_dir,
        gt_dir,
        des_type,
        sv, qe):
    algo_params = {
        'des_type': des_type,
        'max_keypoints': MAX_KEYPOINTS,
        'K': K,
        'L': L,
    }

    results_file = str(Path(cbir.BASE_DIR) / 'results'
                       / '{train_dir}_{test_dir}_{des_type}_{sv}_{qe}.txt'.format(train_dir=str(Path(train_dir).name),
                                                                                  test_dir=str(Path(test_dir).name),
                                                                                  des_type=des_type,
                                                                                  sv=sv,
                                                                                  qe=qe))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'results')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'results'))

    mAP, mAP_new = do_train_test(
        train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir,
        algo_params=algo_params,
        sv=sv,
        qe=qe, )

    info = f'{(mAP, mAP_new)}'

    with open(results_file, 'a') as fout:
        print(info, file=fout)


def start_train_test_all_descriptors_and_modes(
        train_dir,
        test_dir,
        gt_dir, ):
    algo_params = {
        'des_type': None,
        'max_keypoints': MAX_KEYPOINTS,
        'K': K,
        'L': L,
    }

    descriptors = [
        'l2net',
        'HardNetAll',
        # 'surf',
        # 'DeepCompare',
        # 'sift'
    ]

    modes_params = {
        'BoW': (False, False),
        'SV': (True, False),
        # 'SV+QE': (True, True),
    }

    results_file = str(Path(cbir.BASE_DIR) / 'results'
                       / '{train_dir}_{test_dir}.txt'.format(train_dir=str(Path(train_dir).name),
                                                             test_dir=str(Path(test_dir).name), ))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'results')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'results'))

    for mode_name, mode_params in modes_params.items():
        for des_type in descriptors:
            algo_params['des_type'] = des_type
            mAP, mAP_new = do_train_test(
                train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir,
                algo_params=algo_params,
                sv=mode_params[0],
                qe=mode_params[1], )

            info = f'{des_type}\t{mode_name}\t{str(Path(train_dir).name)}\t{str(Path(test_dir).name)}\t{(mAP, mAP_new)}'
            with open(results_file, 'a') as fout:
                print(info, file=fout)


def start_test(
        database_name, index_name, database_photos_dir, gt_dir,
        sv, qe):
    results_file = str(Path(cbir.BASE_DIR) / 'results'
                       / '{database_name}_{index_name}_{sv}_{qe}.txt'.format(database_name=database_name,
                                                                             index_name=index_name,
                                                                             sv=sv,
                                                                             qe=qe))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'results')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'results'))

    mAP, mAP_new = evaluate_only(database_name, index_name, database_photos_dir, gt_dir,
                         sv_enable=sv, qe_enable=qe)


    info = f'{(mAP, mAP_new)}'
    with open(results_file, 'a') as fout:
        print(info, file=fout)


if __name__ == "__main__":
    # tree - L 2 {data_buildings_root}
    # ├── Oxford
    # │   ├── gt
    # │   ├── jpg
    # └── Paris
    #     ├── gt
    #     ├── jpg

    is_sample = True
    suffix = "_sample" if is_sample else ""
    paris = 'Paris' + suffix
    oxford = 'Oxford' + suffix
    data_building_root = str(Path(cbir.BASE_DIR) / 'data' / 'Buildings' / 'Original')
    train_dir = str(Path(data_building_root) / paris / 'jpg')
    test_dir = str(Path(data_building_root) / oxford / 'jpg')
    gt_dir = str(Path(data_building_root) / oxford / 'gt')

    start_train_test_all_descriptors_and_modes(
        train_dir,
        test_dir,
        gt_dir,
    )
