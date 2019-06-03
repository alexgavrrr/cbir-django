import logging
import os
import pickle
import re
from pathlib import Path

import cv2
import numpy
from tqdm import tqdm

import cbir
from cbir.cbir_core import CBIRCore
from cbir.legacy_utils import find_image_files


def load_gt(images_path, gt_path, return_queries_names=None):
    all_files_in_gt_path = find_image_files(gt_path, ['txt'])
    ok = [img for img in all_files_in_gt_path if img.find('ok') != -1]
    good = [img for img in all_files_in_gt_path if img.find('good') != -1]
    junk = [img for img in all_files_in_gt_path if img.find('junk') != -1]
    query = [img for img in all_files_in_gt_path if img.find('query') != -1]

    queries = [[] for i in range(5)]
    ok_answers = [[] for i in range(5)]
    good_answers = [[] for i in range(5)]
    junk_answers = [[] for i in range(5)]
    queries_names = [[] for i in range(5)]

    label = None
    if gt_path.find('paris') != -1 or gt_path.find('Paris') != -1:
        label = 'paris'
    elif gt_path.find('oxford') != -1 or gt_path.find('Oxford') != -1:
        label = 'oxford'
    else:
        print('gt_path', gt_path)
        message = 'Looks like gt_path does not contain paris or oxford\n' \
                  'Put paris or oxford or if you use another dataset change code properly.'
        raise ValueError(message)

    for i, q in enumerate(query):
        q_filename = Path(q).name
        n = int(re.sub("[^0-9]", "", q_filename)) - 1

        with open(q, 'r') as f:
            text = f.readline().split()
            prefix = text[0].split("_")[1]
            if label == 'paris':
                f_name = os.path.join(images_path, text[0] + '.jpg')
            elif label == 'oxford':
                f_name = os.path.join(images_path, text[0][5:] + '.jpg')
            x1 = round(float(text[1]))
            y1 = round(float(text[2]))
            x2 = round(float(text[3]))
            y2 = round(float(text[4]))
            img = cv2.imread(f_name, 0)
            if img is None:
                print(f'cv2.imread on `{f_name}` read None')
            else:
                queries[n].append(img[y1:y2, x1:x2])
                queries_names[n].append(f_name)

        with open(ok[i], 'r') as ok_f:
            tmp = []
            for line in ok_f:
                f_name = os.path.join(images_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            ok_answers[n].append(tmp)

        with open(good[i], 'r') as good_f:
            tmp = []
            for line in good_f:
                f_name = os.path.join(images_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            good_answers[n].append(tmp)

        with open(junk[i], 'r') as junk_f:
            tmp = []
            for line in junk_f:
                f_name = os.path.join(images_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            junk_answers[n].append(tmp)

    if return_queries_names:
        return queries, ok_answers, good_answers, junk_answers, queries_names
    else:
        return queries, ok_answers, good_answers, junk_answers


def AP(query_and_gt_images, answer_images, debug=False):
    query = query_and_gt_images[0]
    gt_images = query_and_gt_images[1]
    gt_images_suffixes = [os.path.split(gt_image)[1] for gt_image in gt_images]

    sum_precision = 0.0
    true_positive = 0
    for index_answer_image, answer_image in enumerate(answer_images):
        if os.path.split(answer_image[0][1])[1] in gt_images_suffixes:
            true_positive += 1
            sum_precision += true_positive / float(index_answer_image + 1)

            if debug:
                print("{}/{}".format(true_positive, index_answer_image + 1))
        elif debug:
            print("-")

    return sum_precision / float(len(answer_images)) if len(answer_images) > 0 else 0


def AP_new(query_and_gt_images, answer_images, debug=False):
    query = query_and_gt_images[0]
    gt_images = query_and_gt_images[1]
    gt_images_suffixes = [os.path.split(gt_image)[1] for gt_image in gt_images]

    max_gt_possible = min(len(gt_images), len(answer_images))
    sum_precision = 0.0
    true_positive = 0
    for index_answer_image, answer_image in enumerate(answer_images):
        if os.path.split(answer_image[0][1])[1] in gt_images_suffixes:
            true_positive += 1
            sum_precision += true_positive / float(index_answer_image + 1)

            if debug:
                print("{}/{}".format(true_positive, index_answer_image + 1))
        elif debug:
            print("-")

    return sum_precision / float(max_gt_possible) if len(max_gt_possible) > 0 else 0


def evaluate(train_dir, test_dir, gt_dir,
             algo_params,
             sv_enable=True, qe_enable=True,
             topk=None, n_test_candidates=100, ):
    topk = topk or 5
    def _validate_algo_params(params):
        for key in ['des_type', 'max_keypoints', 'K', 'L']:
            if key not in params.keys():
                raise ValueError(f'Bad algo params: {params}')

    _validate_algo_params(algo_params)

    test_database_name = f'test_database_{str(Path(train_dir).name)}_{str(Path(test_dir).name)}'
    test_index_name = (
        f'index_{algo_params["des_type"]}_{algo_params["max_keypoints"]}'
        f'_{algo_params["K"]}_{algo_params["L"]}')

    created_now = CBIRCore.create_empty_if_needed(
        test_database_name, test_index_name,
        **algo_params, )

    cbir_core = CBIRCore.get_instance(test_database_name, test_index_name)
    cbir_core.set_fd(cbir_core.load_fd())

    if created_now:
        list_paths_to_images_to_train_clusterer = find_image_files(train_dir, cbir.IMAGE_EXTENSIONS, recursive=False)
        list_paths_to_images_to_index = find_image_files(test_dir, cbir.IMAGE_EXTENSIONS, recursive=False)
        cbir_core.compute_descriptors(list_paths_to_images_to_index,
                                      to_index=True,
                                      for_training_clusterer=False)

        cbir_core.compute_descriptors(list_paths_to_images_to_train_clusterer,
                                      to_index=False,
                                      for_training_clusterer=True)
        cbir_core.train_clusterer()

        cbir_core.set_ca(cbir_core.load_ca())
        cbir_core.add_images_to_index()
    else:
        logging.getLogger().info('Index has already been created so we do not rebuild it here')

    cbir_core.set_ca(cbir_core.load_ca())
    cbir_core.set_bow(cbir_core.load_bow())
    cbir_core.set_inv(cbir_core.load_inv())
    cbir_core.set_most_frequent(cbir_core.load_most_frequent())

    queries, ok_answers, good_answers, junk_answers, queries_names = load_gt(test_dir, gt_dir, return_queries_names=True)
    scores = []
    scores_new = []
    answers = []
    for trial in range(5):
        queries_names_trial = queries_names[trial]
        answers_trial = []
        queries_trial = queries[trial]
        ok_answers_trial = ok_answers[trial]
        good_answers_trial = good_answers[trial]
        right_answers_trial = [ok_answers_trial[j] + good_answers_trial[j] for j, _ in enumerate(ok_answers_trial)]

        # queries_gt_trial = list(zip(queries_trial, right_answers_trial))
        for query_now, query_name_now, gt_now in tqdm(zip(queries_trial, queries_names_trial, right_answers_trial)):
            similar_images = cbir_core.search(
                query_now,
                n_candidates=n_test_candidates,
                topk=topk,
                sv_enable=sv_enable,
                qe_enable=qe_enable,
                query_name=query_name_now,
                p_fine_max=None)

            # TODO DEBUG
            print(f'similar_images: {similar_images}')

            scores.append(AP((query_now, gt_now), similar_images))
            scores_new.append(AP_new((query_now, gt_now), similar_images))
            answers_trial.append([query_name_now, [s[0][1] for s in similar_images]])

        answers.append(answers_trial)

    cbir_core.unset_fd()
    cbir_core.unset_ca()

    answers_file = str(Path(cbir.BASE_DIR) / 'answers'
                       / '{des_type}_{sv_enable}_{qe_enable}'
                         '_{train_dir}_{test_dir}.pkl'.format(des_type=algo_params['des_type'],
                                                              sv_enable=sv_enable,
                                                              qe_enable=qe_enable,
                                                              train_dir=str(Path(train_dir).name),
                                                              test_dir=str(Path(test_dir).name), ))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'answers')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'answers'))
    with open(answers_file, 'wb') as fout:
        pickle.dump(answers, fout, pickle.HIGHEST_PROTOCOL)

    mAP = numpy.mean(scores)
    mAP_new = numpy.mean(scores_new)
    print(f'answers: {answers}')
    print(f'scores: {scores}')
    print(f'scores: {scores_new}')
    print(f'mAP: {mAP}')
    print(f'mAP_mew: {mAP_new}')
    return mAP, mAP_new


def evaluate_only(database_name, index_name, database_photos_dir, gt_dir,
                  sv_enable=True, qe_enable=True,
                  topk=None, n_test_candidates=100,
                  p_fine_max=None):
    topk = topk or 5
    cbir_core = CBIRCore.get_instance(database_name, index_name)
    cbir_core.set_fd(cbir_core.load_fd())
    cbir_core.set_ca(cbir_core.load_ca())

    queries, ok_answers, good_answers, junk_answers, queries_names = load_gt(database_photos_dir, gt_dir, return_queries_names=True)
    scores = []
    scores_new = []
    answers = []
    for trial in range(5):
        queries_names_trial = queries_names[trial]
        answers_trial = []
        queries_trial = queries[trial]
        ok_answers_trial = ok_answers[trial]
        good_answers_trial = good_answers[trial]
        right_answers_trial = [ok_answers_trial[j] + good_answers_trial[j] for j, _ in enumerate(ok_answers_trial)]

        # queries_gt_trial = list(zip(queries_trial, right_answers_trial))
        for query_now, query_name_now, gt_now in tqdm(zip(queries_trial, queries_names_trial, right_answers_trial)):
            similar_images = cbir_core.search(
                query_now,
                n_candidates=n_test_candidates,
                topk=topk,
                sv_enable=sv_enable,
                qe_enable=qe_enable,
                query_name=query_name_now,
                p_fine_max=p_fine_max)

            # TODO DEBUG
            print(f'similar_images: {similar_images}')

            scores.append(AP((query_now, gt_now), similar_images))
            scores_new.append(AP_new((query_now, gt_now), similar_images))
            answers_trial.append([query_name_now, [s[0][1] for s in similar_images]])

        answers.append(answers_trial)

    cbir_core.unset_fd()
    cbir_core.unset_ca()

    answers_file = str(Path(cbir.BASE_DIR) / 'answers'
                       / '{sv_enable}_{qe_enable}'
                         '_{database_name}.pkl'.format(sv_enable=sv_enable,
                                                       qe_enable=qe_enable,
                                                       database_name=database_name))
    if not os.path.exists(str(Path(cbir.BASE_DIR) / 'answers')):
        os.mkdir(str(Path(cbir.BASE_DIR) / 'answers'))
    with open(answers_file, 'wb') as fout:
        pickle.dump(answers, fout, pickle.HIGHEST_PROTOCOL)

    mAP = numpy.mean(scores)
    mAP_new = numpy.mean(scores_new)

    print(f'answers: {answers}')
    print(f'scores: {scores}')
    print(f'scores: {scores_new}')
    print(f'mAP: {mAP}')
    print(f'mAP_new: {mAP_new}')
    return mAP, mAP_new


def show_example_ap():
    query_gt = [None, ('p/1', 'p/3', 'p/5', 'p/7')]

    answer = [
        [[0, 'p/3'], None],
        [[0, 'p/2'], None],
        [[0, 'p/7'], None],
    ]

    print(AP(query_gt, answer, debug=True))


def show_example_gt():
    data_building_root = str(Path(cbir.BASE_DIR) / 'data' / 'Buildings' / 'Original')
    test_dir = str(Path(data_building_root) / 'Oxford' / 'jpg')
    gt_dir = str(Path(data_building_root) / 'Oxford' / 'gt')

    queries, ok_answers, good_answers, junk_answers = load_gt(test_dir, gt_dir)
    print(list(map(len, [queries[0], ok_answers[0], good_answers[0], junk_answers[0]])))
    for trial in range(5):
        for index_now in range(11):
            print(f'query_now: {queries[trial][index_now]}')
            print(f'ok_answers_now: {ok_answers[trial][index_now]}')
            print(f'good_answers_now: {good_answers[trial][index_now]}')
            print(f'junk_answers_now: {junk_answers[trial][index_now]}')

            print('_' * 50)
        print(f"{'_' * 50}\n{'_' * 50}")


if __name__ == "__main__":
    print('Example AP')
    show_example_ap()

    print('Example load gt')
    show_example_gt()
