import os
from pathlib import Path

import cbir
from cbir_evaluation.evaluation import evaluate

MAX_KEYPOINTS = 2000
K = 10
L = 4


def do_train_test(train_dir, test_dir, gt_dir,
                  algo_params,
                  sv=True, qe=True, ):
    mAPs = evaluate(train_dir, test_dir, gt_dir,
                    algo_params=algo_params,
                    sv_enable=sv, qe_enable=qe)
    return mAPs


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

    mAPs = do_train_test(
        train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir,
        algo_params=algo_params,
        sv=sv,
        qe=qe, )

    info = f'{mAPs}'

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
        'surf',
        # 'DeepCompare',
        # 'sift'
    ]

    modes_params = {
        'BoW': (False, False),
        'SV': (True, False),
        'SV+QE': (True, True),
    }

    results_file = str(Path(cbir.BASE_DIR) / 'results'
                       / '{train_dir}_{test_dir}.txt'.format(train_dir=str(Path(train_dir).name),
                                                             test_dir=str(Path(test_dir).name), ))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'results')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'results'))

    for mode_name, mode_params in modes_params.items():
        for des_type in descriptors:
            algo_params['des_type'] = des_type
            mAPs = do_train_test(
                train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir,
                algo_params=algo_params,
                sv=mode_params[0],
                qe=mode_params[1], )

            info = f'{des_type}\t{mode_name}\t{str(Path(train_dir).name)}\t{str(Path(test_dir).name)}\t{mAPs}'
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
