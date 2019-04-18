import logging
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import cbir
from cbir import CONFIG
from cbir.legacy_utils import find_image_files, draw_result
from cbir.models.learned_descriptors import HardNetAll_des, HardNetBrown_des, HardNetHPatches_des
from cbir.models.learned_descriptors import L2net_des
from cbir.models.learned_descriptors import SURF, SIFT
from cbir.vocabulary_tree import VocabularyTree

logger = logging.getLogger('cbir.photo_storage_inverted_file')


class CBIR:
    @classmethod
    def get_instance(cls, database, name):
        if not cls.exists(database, name):
            message = (f'CBIR {name} in database {database} does not exist. First create it. For example, by calling '
                       f'`create_empty`.')
            raise ValueError(message)

        if cls.empty(database, name):
            message = f'CBIR {database} {name} is empty'
            logger.warning(message)

        instance = CBIR()
        instance.database = database
        instance.name = name
        return instance

    @classmethod
    def prepare_place_for_database_if_needed(cls, database):
        if not os.path.exists(Path(cbir.DATABASES) / database):
            os.mkdir(Path(cbir.DATABASES) / database)

    @classmethod
    def prepare_place_for_cbir_index_if_needed(cls, database, name):
        if not os.path.exists(Path(cbir.DATABASES) / database / name):
            os.mkdir(Path(cbir.DATABASES) / database / name)

    @classmethod
    def create_empty_if_needed(cls, database, name,
                               des_type, max_keypoints,
                               K, L):
        """
        Creates empty CBIR instance which indexes 0 objects.
        :param database:
        :param name:
        :param des_type:
        :param max_keypoints:
        :param K:
        :param L:
        :return:
        """
        if not cls.exists(database, name):
            cls.prepare_place_for_database_if_needed(database)
            cls.prepare_place_for_cbir_index_if_needed(database, name)
            cls._init_search_structures(database, name,
                                        des_type, max_keypoints, K, L)

    @classmethod
    def _init_search_structures(cls, database, name,
                                des_type, max_keypoints, K, L):
        """
        :param database:
        :param name:
        :param des_type:
        :param max_keypoints:
        :param K:
        :param L:
        """
        params = {}
        params['des_type'] = des_type
        params['max_keypoints'] = max_keypoints
        params['n_words'] = K ** L
        params['idf'] = []
        params['most_frequent'] = []
        params['least_frequent'] = []

        bow = []
        inverted_index = []
        f_names = []
        cls._save(database, name,
                  params,
                  bow,
                  inverted_index,
                  f_names)

        index = []
        cls._save_index(database, name,
                        params['des_type'],
                        index)

        clusterer = []
        cls._save_clusterer(database, name,
                            params['des_type'],
                            clusterer)

    @classmethod
    def _save(cls, database, name,
              params,
              bow,
              inverted_index,
              f_names):

        params['des_path'] = cls.get_des_path(database, name, params['des_type'])

        storage_path = cls.get_storage_path(database, name)
        postfix = '_{}_{}.pkl'.format(params['des_type'], '')
        params['bow_path'] = os.path.join(storage_path, 'bow' + postfix)
        params['inverted_index_path'] = os.path.join(storage_path, 'inverted_index' + postfix)
        params['f_names_path'] = os.path.join(storage_path, 'f_names' + postfix)

        with open(params['bow_path'], 'wb') as f:
            pickle.dump(bow, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(params['inverted_index_path'], 'wb') as f:
            pickle.dump(inverted_index, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(params['f_names_path'], 'wb') as f:
            pickle.dump(f_names, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(params['params_path'], 'wb') as f:
            pickle.dump(params, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_index(cls, database, name,
                    des_type,
                    index):
        with open(cls.get_des_path(database, name, des_type), 'wb') as f:
            pickle.dump({k: (v[0],
                             [p.pt[0] for p in v[1]],
                             [p.pt[1] for p in v[1]],
                             [p.size for p in v[1]])
                         for k, v in index.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_clusterer(cls, database, name,
                        des_type,
                        clusterer):
        with open(cls.get_clusterer_path(database, name, des_type), 'wb') as f:
            pickle.dump(clusterer, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def get_des_path(cls, database, name, des_type):
        postfix_des = '_{}.pkl'.format(des_type)
        return os.path.join(cls.get_storage_path(database, name), 'des' + postfix_des)

    @classmethod
    def get_clusterer_path(cls, database, name, des_type):
        postfix = '_{}_{}.pkl'.format(des_type, '')
        return os.path.join(cls.get_storage_path(database, name), 'clusterer' + postfix)

    def compute_descriptors(self,
                            list_paths_to_images):
        """
        Computes decriptors for given images
        and saves these descriptors in search structures (sqlite).

        :param list_paths_to_images: images for which to compute descriptors
        """
        raise NotImplementedError

    def train_clusterer(self,
                        list_paths_to_images):
        """
        Trains clusterer on images from `list_paths_to_images_train_clusterer`
        and saves it.

        :param list_paths_to_images: images to use to train clusterer
        """
        raise NotImplementedError

    def add_images_to_index(self,
                            list_paths_to_images):
        """
        Adds images to search index.

        :param list_paths_to_images: images to add to index
        """
        raise NotImplementedError

    @classmethod
    def copy_descriptors_from_to(cls, database, from_name, to_name):
        if not cls.descriptors_compatible(database, from_name, to_name):
            message = f"Database {database}'s descriptors {from_name} and {to_name} are not compatible"
            raise ValueError(message)

        raise NotImplementedError

    @classmethod
    def descriptors_compatible(cls, database, first_name, second_name):
        from_des_type = CBIR(database, first_name).get_des_type()
        to_des_type = CBIR(database, second_name).get_des_type()
        return from_des_type != to_des_type

    @classmethod
    def get_databases(cls):
        return [
            filename
            for filename in os.listdir(cbir.DATABASES)
            if os.path.isdir(Path(cbir.DATABASES) / filename)]

    @classmethod
    def get_cbir_indexes_of_database(cls, database):
        return [
            filename
            for filename in os.listdir(Path(cbir.DATABASES) / database)
            if os.path.isdir(Path(cbir.DATABASES) / database / filename)]

    @classmethod
    def get_storage_path(cls, database, name):
        return str(Path(cbir.DATABASES) / database / name)

    @classmethod
    def exists(cls, database, name):
        return (database in CBIR.get_databases()
                and name in CBIR.get_cbir_indexes_of_database(database)
                and cls._inited_properly(database, name))

    @classmethod
    def _inited_properly(cls, database, name):
        raise NotImplementedError

    @classmethod
    def empty(cls, database, name):
        raise NotImplementedError

    def __str__(self):
        return f'CBIR {self.database} {self.name}'

    def get_des_type(self):
        raise NotImplementedError

    def search(self,
               list_paths_to_images,
               n_candidates=100,
               topk=5, n_inliners_thr=20, max_verified=20,
               qe_avg=50, qe_limit=30, new_query=None,
               sv_enable=True, qe_enable=True, debug=False):
        """
        :param list_paths_to_images: query images
        :return: list paths images most similar to the query
        """
        raise NotImplementedError


# database = 'first'
# name = 'index_first'
#
# des_type = 'l2net'
# max_keypoints = 2000
# K = 10
# L = 2
#
# list_paths_to_images_to_train_clusterer = []
# list_paths_to_images_to_index = []
#
# CBIR.create_empty_if_needed(database, name,
#                             des_type=des_type, max_keypoints=max_keypoints,
#                             K=K, L=L)
# cbir_index = CBIR.get_instance(database, name)
# cbir_index.compute_descriptors(list(set(list_paths_to_images_to_index)
#                                     | set(list_paths_to_images_to_train_clusterer)))
# cbir_index.train_clusterer(list_paths_to_images_to_train_clusterer)
# cbir_index.add_images_to_index(list_paths_to_images_to_index)
#
# query_images = []
# result_images = cbir_index.search(query_images)


class Storage:
    """
        Creates new storage from a given
        directory with photos.
    """

    def __init__(self,
                 storage_path, testing_path, training_path,
                 des_type,
                 label="", max_keypoints=500, L=5, K=10,
                 extensions=['jpg'],
                 debug=False):
        self.storage_path = storage_path
        self.testing_path = testing_path
        self.training_path = training_path
        self.max_keypoints = max_keypoints
        self.K = K
        self.L = L
        self.n_words = self.K ** self.L
        self.des_type = des_type
        self.extensions = extensions
        self.label = label
        self.most_frequent = None
        self.least_frequent = None
        self.bow = None
        self.ca = None
        self.fd = None
        self.idf = None
        self.f_names = None
        self.index = None
        self.inverted_index = None
        self.debug = debug

        # Init fd
        if des_type == 'sift':
            self.fd = SIFT(max_keypoints=max_keypoints)
        elif des_type == 'surf':
            self.fd = SURF(max_keypoints=max_keypoints)
        elif des_type == 'l2net':
            self.fd = L2net_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif des_type == 'DeepCompare':
            message = 'DeepCompare is not supported anymore'
            raise ValueError(message)
        elif des_type == 'HardNetBrown':
            self.fd = HardNetBrown_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif des_type == 'HardNetAll':
            self.fd = HardNetAll_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif des_type == 'HardNetHPatches':
            self.fd = HardNetHPatches_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        else:
            self.fd = None

        postfix_des = '_{}.pkl'.format(self.des_type)
        postfix = '_{}_{}.pkl'.format(self.des_type, label)

        self.des_path = 'des' + postfix_des
        self.des_path = os.path.join(self.storage_path, self.des_path)

        self.bow_path = 'bow' + postfix
        self.bow_path = os.path.join(self.storage_path, self.bow_path)

        self.ca_path = 'clusterer' + postfix
        self.ca_path = os.path.join(self.storage_path, self.ca_path)

        self.inverted_index_path = 'inverted_index' + postfix
        self.inverted_index_path = os.path.join(self.storage_path, self.inverted_index_path)

        self.f_names_path = 'f_names' + postfix
        self.f_names_path = os.path.join(self.storage_path, self.f_names_path)

        self.params_path = 'params' + postfix
        self.params_path = os.path.join(self.storage_path, self.params_path)

        if (os.path.exists(self.inverted_index_path)
                and os.path.exists(self.f_names_path)
                and os.path.exists(self.params_path)
                and os.path.exists(self.ca_path)
                and os.path.exists(self.des_path)
                and os.path.exists(self.bow_path)):
            self.load()
            self.load_ca()
            self.load_descriptors()
        else:
            start = time.time()
            self.create_new_index()
            self.save()
            print("Index has been created in {} m".format((time.time() - start) / float(60)))

    def create_new_index(self, max_training_images=None):
        """
        Creates new index file with correspondence
        between photos and their descriptors.
        """
        index = {}
        descriptors = []

        index_ready = os.path.exists(self.des_path)
        ca_ready = os.path.exists(self.ca_path)

        if os.path.abspath(self.storage_path) == os.path.abspath(self.training_path):

            start = time.time()
            # images = list(random.sample(find_image_files(self.dir, self.extensions), 4000))
            images = find_image_files(self.testing_path, self.extensions)
            print("List of file names created in {} s".format(time.time() - start))

            start = time.time()
            if not index_ready:
                for f in tqdm(images, total=len(images)):
                    tmp = self.get_descriptor(f, raw=True)
                    if tmp[0] is not None:
                        index[f] = tmp
                        descriptors.append(tmp[0])
                    else:
                        print("No keypoints found for {}".format(f))

                print("Descriptors calculated in {} m".format((time.time() - start) / float(60)))

                self.index = index
                self.save_descriptors()
                del index
                del self.index
                self.index = None
                index = None
            else:
                self.load_descriptors()
                descriptors = [v[0] for k, v in self.index.items()]
                del self.index
                self.index = None

            descriptors = np.concatenate(descriptors, axis=0)

            print("All descriptors:", descriptors.shape)
            print(f"Size in memory: {descriptors.shape[0] * descriptors.shape[1] * 4 / float(1024 ** 2)} MB")

            print("Constructing bag of words for image descriptors...")

            start = time.time()
            if ca_ready:
                self.load_ca()
            else:
                self.ca = VocabularyTree(L=self.L, K=self.K).fit(descriptors)
                print(f"Vocabulary constructed in {(time.time() - start) / float(60)} m")
            del descriptors
            self.load_descriptors()
        else:
            if not ca_ready:
                training_des = []

                start = time.time()
                if max_training_images is None:
                    training_images = find_image_files(self.training_path,
                                                       self.extensions)[:max_training_images]
                else:
                    training_images = find_image_files(self.training_path,
                                                       self.extensions)
                print("List of file names created in {} s".format(time.time() - start))

                start = time.time()
                training_des_path = os.path.join(self.training_path,
                                                 'des_{}.pkl'.format(self.des_type))

                if os.path.exists(training_des_path):
                    print("Loading training descriptors...")
                    with open(training_des_path, 'rb') as f:
                        training_des = pickle.load(f)
                        training_des = [v[0] for k, v in training_des.items()]
                else:
                    tmp_index = {}
                    for f in tqdm(training_images, total=len(training_images)):
                        tmp = self.get_descriptor(f, raw=True)
                        if tmp[0] is not None:
                            tmp_index[f] = tmp
                            training_des.append(tmp[0])
                        else:
                            print("No keypoints found for {}".format(f))

                    print("Saving trainig descriptors...")
                    with open(training_des_path, 'wb') as f:
                        pickle.dump({k: (v[0],
                                         [p.pt[0] for p in v[1]],
                                         [p.pt[1] for p in v[1]],
                                         [p.size for p in v[1]])
                                     for k, v in tmp_index.items()}, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    del tmp_index

                training_des = np.concatenate(training_des, axis=0)

                print("Training descriptors calculated in {} m".format((time.time() - start) / float(60)))
                print("All descriptors[training]:", training_des.shape)
                print("Size in memory[training]:", training_des.shape[0] * training_des.shape[1] * 4 / float(1024 ** 2))

                print("Constructing bag of words for image descriptors...")
                start = time.time()
                self.ca = VocabularyTree(L=self.L, K=self.K).fit(training_des)
                del training_des
                del training_images
                print("Vocabulary constructed in {} m".format((time.time() - start) / float(60)))
            else:
                self.load_ca()

            if not index_ready:
                images = find_image_files(self.testing_path, self.extensions)
                start = time.time()
                for f in tqdm(images, total=len(images)):
                    tmp = self.get_descriptor(f, raw=True)
                    if tmp[0] is not None:
                        index[f] = tmp
                    else:
                        print("No keypoints found for {}".format(f))
                print("Descriptors calculated in {} m".format((time.time() - start) / float(60)))
                self.index = index
                self.save_descriptors()
            else:
                self.load_descriptors()

        if not ca_ready:
            self.save_ca()

        print("Constructing corpus(BoVW representation)")

        # TODO: Ensure that this snippet for initializing corpus and f_names is faster and better.
        # corpus = [self.ca.predict(value[0]) for value in tqdm(self.index.values(),
        #                                                       total=len(self.index))]
        # self.f_names = list(self.index.keys())
        corpus = [(self.ca.predict(value[0]), key) for key, value in tqdm(self.index.items(),
                                                                          total=len(self.index))]
        corpus = list(zip(*corpus))
        corpus, self.f_names = corpus

        print("Creating inverted file and calculate tf-idf weights")
        self.bow = np.zeros((len(self.f_names), self.n_words + 1), dtype=np.int16)
        self.inverted_index = [set() for i in range(self.n_words)]

        freqs = np.zeros(self.n_words, dtype=np.int16)

        for i, image in enumerate(corpus):
            flags = np.zeros(self.n_words, dtype=np.int8)
            for word in image:
                self.bow[i][word] += 1
                self.bow[i][self.n_words] += 1
                if flags[word] == 0:
                    freqs[word] += 1
                    flags[word] = 1
                self.inverted_index[word].add(i)

        freqs = np.argsort(freqs)
        five_percent = int(0.085 * self.n_words)
        self.most_frequent = freqs[-five_percent:]
        self.least_frequent = freqs[:five_percent]

        self.idf = compute_idf(self.bow)
        self.bow = csr_matrix(self.bow)

    def get_bow_vector(self, img_des):
        res = np.zeros(self.n_words)

        if img_des.ndim == 1:
            img_des = img_des.reshape(img_des.shape[0], -1)

        pred = self.ca.predict(img_des)
        for p in pred:
            res[p] += 1

        return res / float(img_des.shape[0])

    def get_candidates(self, query, filter=True):
        """

        :param query:  `[(visual_word, freq), ...]` query as a bow(BoVW)
        :param filter:
        :return:
        """
        candidates = set()

        for i, freq in enumerate(query):
            # Don't pay attention to frequent words
            if freq == 0 or (filter and i in self.most_frequent):
                continue
            candidates = candidates | self.inverted_index[i]

        return candidates

    def ransac(self, img_descriptor, kp, candidates,
               min_inliners, max_verified):

        descriptors = [self.index[img[1]] for img in candidates]

        descriptors, descriptors_kp = list(zip(*descriptors))

        # Brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2)

        MIN_MATCH_COUNT = 10
        all_matches = []
        all_matches_masks = []
        all_transforms = []

        verified = []
        n_verified = 0

        for i, des in enumerate(descriptors):
            matches = bf.match(img_descriptor, des)
            if len(matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([descriptors_kp[i][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

            else:
                print("Not enough matches are found - {}/{}".format(len(matches),
                                                                    MIN_MATCH_COUNT))
                matchesMask = None

            all_matches.append(matches)
            all_matches_masks.append(matchesMask)
            all_transforms.append(M)

            if np.count_nonzero(matchesMask) > min_inliners:
                verified.append(1)
                n_verified += 1
            else:
                verified.append(0)

            if n_verified == max_verified:
                break

        return [all_matches, all_matches_masks, all_transforms,
                verified,
                descriptors_kp]

    def get_similar(self, img_path, n_candidates=100,
                    topk=5, n_inliners_thr=20, max_verified=20,
                    qe_avg=50, qe_limit=30, new_query=None,
                    sv_enable=True, qe_enable=True, debug=False):

        start = time.time()
        # STEP 1. APPLY INVERTED INDEX TO GET CANDIDATES

        result_tuple = self.get_descriptor(img_path, both=True)
        if len(result_tuple) != 3 or result_tuple[0] is None:
            message = f'Could not get descriptor for image {img_path}'
            raise ValueError(message)
        img_descriptor, img_bovw, kp = result_tuple

        print("Descriptors got in {}".format(time.time() - start))
        if debug:
            pass

        start = time.time()

        if new_query is not None:
            img_bovw = new_query[:-1].astype(np.float32) / new_query[-1]

        candidates = self.get_candidates(img_bovw)
        if len(candidates) == 0:
            candidates = self.get_candidates(img_bovw, filter=False)
        # print(len(candidates))
        print("Candidates got in {}".format(time.time() - start))
        if debug:
            print(len(candidates))

        # STEP 2. PRELIMINARY RANKING
        start = time.time()

        ranks = []
        for i in candidates:
            bow = np.array(self.bow[i].todense(), dtype=np.float).flatten()
            ranks.append((i, euclidean(img_bovw * self.idf, bow[:-1] / bow[-1] * self.idf)))

        ranks = sorted(ranks, key=lambda x: x[1])
        candidates = [(i[0], self.f_names[i[0]]) for i in ranks[:n_candidates]]

        print("Short list got in {}".format(time.time() - start))
        if debug:
            pass

        # STEP 3. SPATIAL VERIFICATION
        if not sv_enable:
            # for compatibility with the output format
            candidates = [(c, 0) for c in candidates]
            return candidates[:topk]

        start = time.time()
        [all_matches, all_matches_masks, all_transforms,
         verified,
         descriptors_kp] = self.ransac(img_descriptor, kp, candidates,
                                       n_inliners_thr, max_verified)
        print('Spatial verification got in {}s'.format(time.time() - start))
        if debug:
            n = 1
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=all_matches_masks[n],
                               flags=0)

            if type(img_path) is str:
                img1 = cv2.imread(img_path, 0)
            else:
                img1 = img_path
            img2 = cv2.imread(candidates[n][1])

            h, w = img1.shape
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, all_transforms[n])
            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, (255, 50, 255), 30, cv2.LINE_AA)

            img3 = cv2.drawMatches(img1, kp,
                                   img2, descriptors_kp[n],
                                   all_matches[n], None, **draw_params)
            print(img3.shape)
            img3[:, :, 0], img3[:, :, 2] = img3[:, :, 2], img3[:, :, 0]
            plt.imshow(img3)
            print(all_transforms)
            plt.show()

        sv_candidates = sorted([(candidates[i],
                                 np.count_nonzero(all_matches_masks[i]))
                                for i in range(len(verified))],
                               key=lambda x: x[1], reverse=True)

        # STEP 4. QUERY EXPANSION
        if qe_enable and new_query is None and np.count_nonzero(verified) < qe_limit:
            start = time.time()
            top_res = []
            for r in sv_candidates[:qe_avg]:
                bow = np.array(self.bow[r[0][0]].todense()).flatten()
                top_res.append(bow)
            one_more_query = (sum(top_res) + img_bovw) / (len(top_res) + 1)

            new_sv_candidates = self.get_similar(img_path, n_candidates,
                                                 topk, n_inliners_thr, max_verified,
                                                 qe_avg, qe_limit, one_more_query,
                                                 qe_enable=qe_enable, sv_enable=sv_enable,
                                                 debug=debug)

            old = set(sv_candidates[i][0][0] for i in range(len(sv_candidates)))
            new = set(new_sv_candidates[i][0][0] for i in range(len(new_sv_candidates)))
            dublicates = set(old) & set(new)

            new_sv_candidates = [el for el in new_sv_candidates
                                 if el[0][0] not in dublicates]
            sv_candidates = sorted(sv_candidates + new_sv_candidates,
                                   key=lambda x: x[1], reverse=True)

        if debug and qe_enable and new_query is None:
            print("Query Expansion got in {}s".format(time.time() - start))

        return sv_candidates[:topk]

    def get_descriptor(self, img_path, raw=False, both=False):
        """
        Calculate descriptor for the specified image.
        """
        if type(img_path) is str:
            img = cv2.imread(img_path, 0)
        elif type(img_path) is np.ndarray:
            img = img_path
        else:
            print("Unknown type")
            img = None

        if img is None or self.fd is None:
            return None, None

        kp = self.fd.detect(img, None)
        kp, des = self.fd.compute(img, kp)

        if kp is None:
            return None, None

        des = np.array(des)

        if both and self.ca is not None:
            return des, self.get_bow_vector(des), kp

        if raw or self.ca is None:
            return des, kp
        else:
            return self.get_bow_vector(des), kp

    def load_descriptors(self):
        print("Loading descriptors...")
        with open(self.des_path, 'rb') as f:
            self.index = pickle.load(f)
        self.index = {k: (v[0],
                          [cv2.KeyPoint(*el) for el in zip(v[1], v[2], v[3])])
                      for k, v in self.index.items()}

    def save_descriptors(self):
        print("Saving descriptors...")
        with open(self.des_path, 'wb') as f:
            pickle.dump({k: (v[0],
                             [p.pt[0] for p in v[1]],
                             [p.pt[1] for p in v[1]],
                             [p.size for p in v[1]])
                         for k, v in self.index.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_ca(self):
        print("Loading clusterer...")
        with open(self.ca_path, 'rb') as f:
            self.ca = pickle.load(f)

    def save_ca(self):
        print("Saving clusterer...")
        with open(self.ca_path, 'wb') as f:
            pickle.dump(self.ca, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(os.path.join(self.storage_path,
                               'params_{}_{}.pkl'.format(self.des_type,
                                                         self.label)), 'rb') as f:
            params = pickle.load(f)

        self.max_keypoints = params['max_keypoints']
        self.n_words = params['n_words']
        self.idf = params['idf']

        self.ca_path = params['ca_path']
        self.des_path = params['des_path']
        self.bow_path = params['bow_path']
        self.inverted_index_path = params['inverted_index_path']
        self.f_names_path = params['f_names_path']

        with open(self.bow_path, 'rb') as f:
            self.bow = pickle.load(f)
        with open(self.inverted_index_path, 'rb') as f:
            self.inverted_index = pickle.load(f)
        with open(self.f_names_path, 'rb') as f:
            self.f_names = pickle.load(f)

        if 'most_frequent' not in params.keys():
            freqs = np.count_nonzero(self.bow, axis=0)
            print(np.mean(freqs), np.std(freqs), np.max(freqs), np.median(freqs))
            freqs = np.argsort(freqs)
            part_to_drop = int(0.085 * self.n_words)
            self.most_frequent = freqs[-part_to_drop:]
            self.least_frequent = freqs[:part_to_drop]
            params['most_frequent'] = self.most_frequent
            params['least_frequent'] = self.least_frequent
            with open(self.params_path, 'wb') as f:
                pickle.dump(params, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.most_frequent = params['most_frequent']
            self.least_frequent = params['least_frequent']

    def save(self):
        with open(self.bow_path, 'wb') as f:
            pickle.dump(self.bow, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.inverted_index_path, 'wb') as f:
            pickle.dump(self.inverted_index, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.f_names_path, 'wb') as f:
            pickle.dump(self.f_names, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        params = dict()
        params['max_keypoints'] = self.max_keypoints
        params['n_words'] = self.n_words
        params['idf'] = self.idf
        params['most_frequent'] = self.most_frequent
        params['least_frequent'] = self.least_frequent
        params['bow_path'] = self.bow_path
        params['inverted_index_path'] = self.inverted_index_path
        params['f_names_path'] = self.f_names_path
        params['ca_path'] = self.ca_path
        params['des_path'] = self.des_path

        with open(self.params_path, 'wb') as f:
            pickle.dump(params, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def show(self):
        print('a')


def compute_idf(bow):
    idf = np.zeros(bow.shape[1] - 1)
    for word in range(bow.shape[1] - 1):
        s = np.sum(bow[:, word] > 0)
        if s != 0:
            idf[word] = np.log(bow.shape[0] / float(s))
    return idf


def main(storage_path, training_path, label):
    test_path = storage_path
    storage = Storage(storage_path,
                      test_path,
                      training_path,
                      'l2net',
                      label=label,
                      max_keypoints=2000,
                      L=2,
                      K=10,
                      extensions=['jpg'],
                      debug=True)
    while True:
        print("Enter command")
        cmd = input().split()
        if cmd[0] == 's' and len(cmd) == 4:
            similar = storage.get_similar(cmd[1], topk=int(cmd[2]),
                                          n_candidates=int(cmd[3]), debug=True,
                                          qe_enable=False)
            print(similar)
            draw_result(cmd[1],
                        [v[1] for v in list(zip(*similar))[0]])
        elif cmd[0] == 'q':
            break
        else:
            print("Unknown command")

    print("Bye!")


if __name__ == "__main__":
    main(
        '/Users/alexgavr/main/Developer/Data/Buildings/Revisited/datasets/roxford5k_sample/jpg',
        '/Users/alexgavr/main/Developer/Data/Buildings/Revisited/datasets/roxford5k_sample/jpg',
        'First Label')
