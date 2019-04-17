import os
import pickle
import re
from pathlib import Path

import cv2
import numpy
from tqdm import tqdm

from cbir.legacy_utils import find_image_files
from cbir.photo_storage_inverted_file import Storage


def load_gt(img_path, gt_path):
    all_files_in_gt_path = find_image_files(gt_path, ['txt'])
    ok = [img for img in all_files_in_gt_path if img.find('ok') != -1]
    good = [img for img in all_files_in_gt_path if img.find('good') != -1]
    junk = [img for img in all_files_in_gt_path if img.find('junk') != -1]
    query = [img for img in all_files_in_gt_path if img.find('query') != -1]

    queries = [[] for i in range(5)]
    ok_answers = [[] for i in range(5)]
    good_answers = [[] for i in range(5)]
    junk_answers = [[] for i in range(5)]

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
                f_name = os.path.join(img_path, prefix, text[0] + '.jpg')
            elif label == 'oxford':
                f_name = os.path.join(img_path, text[0][5:] + '.jpg')
            x1 = round(float(text[1]))
            y1 = round(float(text[2]))
            x2 = round(float(text[3]))
            y2 = round(float(text[4]))
            img = cv2.imread(f_name, 0)
            if img is None:
                print(f'cv2.imread on `{f_name}` read None')
            else:
                queries[n].append(img[y1:y2, x1:x2])

        with open(ok[i], 'r') as ok_f:
            tmp = []
            for line in ok_f:
                f_name = os.path.join(img_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            ok_answers[n].append(tmp)

        with open(good[i], 'r') as good_f:
            tmp = []
            for line in good_f:
                f_name = os.path.join(img_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            good_answers[n].append(tmp)

        with open(junk[i], 'r') as junk_f:
            tmp = []
            for line in junk_f:
                f_name = os.path.join(img_path, line[:-1] + '.jpg')
                tmp.append(f_name)
            junk_answers[n].append(tmp)

    return queries, ok_answers, good_answers, junk_answers


def AP(query, similar, debug=False):
    score = 0.0
    rel = 0
    for i, s in enumerate(similar):
        for gt in query[1]:
            if os.path.split(s[0][1])[1] == os.path.split(gt)[1]:
                rel += 1
                score += rel / float(i + 1)
                if debug:
                    print("{}/{}".format(rel, i + 1))
                break
    return score / float(len(similar))


def evaluate(test_dir, train_dir, des_type, gt_dir, label,
             topk=5, n_test=100, sv_enable=True, qe_enable=True):
    storage_dir = test_dir
    storage = Storage(storage_dir, test_dir, train_dir, des_type, label=label,
                      max_keypoints=2000, L=5, K=10)
    queries, ok_answers, good_answers, junk_answers = load_gt(test_dir, gt_dir)
    scores = []
    results = []

    for i in range(5):
        res = []
        i_q = queries[i]
        i_ok = ok_answers[i]
        i_good = good_answers[i]
        right_answers = [i_ok[j] + i_good[j] for j, _ in enumerate(i_ok)]
        i_q = list(zip(i_q, right_answers))
        for q in tqdm(i_q):
            similar = storage.get_similar(q[0], n_test, topk,
                                          sv_enable=sv_enable,
                                          qe_enable=qe_enable)
            scores.append(AP(q, similar))
            res.append([q[0], [s[0][1] for s in similar]])
        results.append(res)

    output_file = './answers/{}_{}_{}_{}.pkl'.format(des_type,
                                                     label,
                                                     sv_enable,
                                                     qe_enable)
    if not os.path.exists('./answers'):
        os.mkdir('./answers')

    with open(output_file, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(scores)
    mAP = numpy.mean(scores)
    return mAP


def show_example_ap():
    answer = [[[0, 'p/3']], [[0, 'p/7']], [[0, 'p/2']]]
    query = [None, ('p/1', 'p/3', 'p/5', 'p/7')]
    print(AP(query, answer, debug=True))


def show_example_gt():
    data_buildings_root = os.environ.get('DATA_BUILDINGS_ROOT') or '~/main/data'
    test_dir = os.path.join(data_buildings_root, 'Paris_sample', 'jpg')
    gt_dir = os.path.join(data_buildings_root, 'Paris_sample', 'gt')

    queries, ok_answers, good_answers, junk_answers = load_gt(test_dir, gt_dir)
    # print(queries)
    print('\n\n'.join(map(lambda lst: '\n'.join(map(str, lst)), ok_answers[0])))


if __name__ == "__main__":
    show_example_ap()
    show_example_gt()
