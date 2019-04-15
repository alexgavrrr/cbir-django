import os

from cbir_tests.evaluation import evaluate


def train_my_test_my(des_type, output_file, sv=True, qe=True,
                     train_dir=None, test_dir=None, gt_dir=None, is_sample=False):
    data_buildings_root = os.environ.get('DATA_BUILDINGS_ROOT') or '~/main/data'
    # tree - L 2 {data_buildings_root}
    # ├── Oxford
    # │   ├── gt
    # │   ├── gt_files_170407.tgz
    # │   ├── jpg
    # │   └── oxbuild_images.tgz
    # ├── Oxford_sample
    # │   └── jpg
    #     └── gt
    # └── Paris
    #     ├── gt
    #     ├── jpg
    #     └── paris_120310.tgz
    # └── Paris_sample
    #     ├── gt
    #     └── jpg

    if (train_dir or test_dir or gt_dir) and is_sample:
        message = 'is_sample is not compatible with non-empty train_dir or test_dir or gt_dir'
        raise ValueError(message)

    if is_sample:
        train_dir = os.path.join(data_buildings_root, 'Oxford_sample', 'jpg')
        test_dir = os.path.join(data_buildings_root, 'Paris_sample', 'jpg')
        gt_dir = os.path.join(data_buildings_root, 'Paris_sample', 'gt')
    else:
        train_dir = train_dir or os.path.join(data_buildings_root, 'Oxford', 'jpg')
        test_dir = test_dir or os.path.join(data_buildings_root, 'Paris', 'jpg')
        gt_dir = gt_dir or os.path.join(data_buildings_root, 'Paris', 'gt')

    mAP = evaluate(test_dir, train_dir, des_type,
                   gt_dir, "oxford",
                   sv_enable=sv, qe_enable=qe)

    ans = f'{des_type} trained on {train_dir} got {mAP} mAP on {test_dir}'

    with open(output_file, 'a') as f:
        print(ans, file=f)

    return ans


def do_train_test(train_dir, test_dir, gt_dir, is_sample):
    descriptors = [
        'l2net',
        # 'HardNetAll',
        # 'surf',
        # 'DeepCompare',
        # 'sift'
    ]

    modes_params = {
        'BoW': (False, False),
        # 'SV': (True, False),
        # 'SV+QE': (True, True),
    }

    for mode_name, mode_params in modes_params.items():
        OUTPUT_FILE = 'results_{}.txt'.format(mode_name)

        with open(OUTPUT_FILE, 'a') as f:
            print("####################", file=f)
            for des_type in descriptors:
                train_my_test_my(des_type, OUTPUT_FILE, *mode_params,
                                 train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir, is_sample=is_sample)


if __name__ == "__main__":
    do_train_test(None, None, None, True)
