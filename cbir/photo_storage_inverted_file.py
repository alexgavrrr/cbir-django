import logging
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import cbir
from cbir import CONFIG
from cbir.legacy_utils import draw_result
from cbir.models.learned_descriptors import HardNetAll_des, HardNetBrown_des, HardNetHPatches_des
from cbir.models.learned_descriptors import L2net_des
from cbir.models.learned_descriptors import SURF, SIFT
from cbir.vocabulary_tree import VocabularyTree

logger = logging.getLogger('cbir.photo_storage_inverted_file')

DES_TYPE = 'l2net'
MAX_KEYPOINTS = 2000


# TODO:
# Now all images in the index are both for training clusterer and for indexing.
# In the future distinguishing between purposes will be required.

class CBIR:
    @classmethod
    def get_instance(cls, database, name):
        if not cls.exists(database, name):
            message = (f'CBIR {name} in database {database} does not exist. First create it. For example, by calling '
                       f'`create_empty`.')
            raise ValueError(message)

        instance = CBIR()
        instance.database = database
        instance.name = name

        params = instance.load_params()
        instance.des_type = params['des_type']
        instance.max_keypoints = params['max_keypoints']
        instance.K = params['K']
        instance.L = params['L']
        instance.n_words = instance.K ** instance.L

        instance.fd = None
        instance.ca = None

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
        params['K'] = K
        params['L'] = L
        params['n_words'] = K ** L

        data_dependent_params = {}
        data_dependent_params['idf'] = []
        data_dependent_params['most_frequent'] = []
        data_dependent_params['least_frequent'] = []

        index = {}
        clusterer = []
        bow = None
        inverted_index = [set() for i in range(params['n_words'])]
        freqs = np.zeros(params['n_words'], dtype=np.int16)
        f_names = []

        cls._save_params(database, name, params)
        cls._save_data_dependent_params(database, name, params)
        cls._save_index(database, name, index)
        cls._save_clusterer(database, name, clusterer)
        cls._save_bow(database, name, bow)
        cls._save_inverted_index(database, name, inverted_index)
        cls._save_freqs(database, name, freqs)
        cls._save_f_names(database, name, f_names)

    def compute_descriptors(self,
                            list_paths_to_images):
        """
        Computes decriptors for given images
        and saves these descriptors in search structures (sqlite).

        :param list_paths_to_images: images for which to compute descriptors
        """
        fd_loaded_before = self.fd is not None
        if not fd_loaded_before:
            self.set_fd(self.load_fd())

        # TODO: Memory. Loading full index. Fix it.
        index = self.load_index()

        count_new = 0
        count_defects = 0
        count_old = 0
        for path_to_image in list_paths_to_images:
            print(path_to_image)
            if path_to_image in index.keys():
                count_old += 1
            else:
                tmp = self.get_descriptor(path_to_image, raw=True)
                if tmp[0] is not None:
                    count_new += 1
                    index[path_to_image] = tmp
                else:
                    count_defects += 1
                    print("No keypoints found for {}".format(path_to_image))

        if not fd_loaded_before:
            self.unset_fd()

        print(f'count_new: {count_new} ; count_defects: {count_defects} ; count_old: {count_old}')
        CBIR._save_index(self.database, self.name, index)

    def get_descriptor(self, path_to_image, raw=False, both=False, total_count_coordinate_for_bow=True):
        if type(path_to_image) is str:
            image = cv2.imread(path_to_image, 0)
        elif type(path_to_image) is np.ndarray:
            image = path_to_image
        else:
            print("Unknown type")
            image = None

        if image is None or self.fd is None:
            return None, None

        kp = self.fd.detect(image, None)
        kp, des = self.fd.compute(image, kp)

        if kp is None:
            return None, None

        des = np.array(des)

        if both and self.ca is not None:
            return des, self.get_bow_vector(des, total_count_coordinate_for_bow), kp

        if raw or self.ca is None:
            return des, kp
        else:
            return self.get_bow_vector(des, total_count_coordinate_for_bow), kp

    def get_bow_vector(self, img_des, total_count=True):
        res = np.zeros(self.n_words
                       if not total_count
                       else self.n_words + 1)

        if img_des.ndim == 1:
            img_des = img_des.reshape(img_des.shape[0], -1)
        pred = self.ca.predict(img_des)

        for p in pred:
            res[p] += 1

        if total_count:
            res[self.n_words] = len(pred)

        return res / float(img_des.shape[0])

    def train_clusterer(self,
                        list_paths_to_images):
        """
        Trains clusterer on images from `list_paths_to_images_train_clusterer`
        and saves it.

        :param list_paths_to_images: images to use to train clusterer
        """
        # TODO: Handle case if some photo from list_paths_to_images not in index yet.

        # TODO: Memory. Loading full index. Fix it.
        index = self.load_index()
        descriptors_for_training = [v[0] for k, v in index.items() if k in list_paths_to_images]
        descriptors_for_training = np.concatenate(descriptors_for_training, axis=0)

        ca = VocabularyTree(L=self.L, K=self.K).fit(descriptors_for_training)
        CBIR._save_clusterer(self.database, self.name, ca)

    def add_images_to_index(self,
                            list_paths_to_images):
        """
        Adds images to search index.

        :param list_paths_to_images: images to add to index
        """

        # TODO: Memory. Loading full index. Fix it.
        index = self.load_index()

        ca_loaded_before = self.ca is not None
        if not ca_loaded_before:
            self.set_ca(self.load_ca())

        corpus = [(self.ca.predict(value[0]), key)
                  for key, value
                  in tqdm(index.items(), total=len(index))
                  if key in list_paths_to_images
                  ]

        if not ca_loaded_before:
            self.unset_ca()

        corpus = list(zip(*corpus))
        corpus, f_names_new = corpus

        # TODO: Memory. Bow can be bery big. Consider about it.
        bow_new = np.zeros((len(f_names_new), self.n_words + 1), dtype=np.int16)
        freqs = self.load_freqs()

        # TODO: Memory. inverted_index can be bery big. Consider about it.
        inverted_index = self.load_inverted_index()
        for i, image in enumerate(corpus):
            for word in image:
                bow_new[i][word] += 1
                bow_new[i][self.n_words] += 1
                if bow_new[i][word] == 1:
                    freqs[word] += 1
                inverted_index[word].add(i)

        bow = self.load_bow()
        bow_new = sparse.vstack((bow, bow_new), format='csr')

        f_names = self.load_f_names()
        f_names_new = f_names + list(f_names_new)

        five_percent = int(0.085 * self.n_words)
        freqs = np.argsort(freqs)
        most_frequent = freqs[-five_percent:]
        least_frequent = freqs[:five_percent]
        idf = compute_idf(bow_new.todense())

        data_dependent_params = {}
        data_dependent_params['idf'] = idf
        data_dependent_params['most_frequent'] = most_frequent
        data_dependent_params['least_frequent'] = least_frequent

        CBIR._save_data_dependent_params(self.database, self.name, data_dependent_params)
        CBIR._save_bow(self.database, self.name, bow_new)
        CBIR._save_inverted_index(self.database, self.name, inverted_index)
        CBIR._save_freqs(self.database, self.name, freqs)
        CBIR._save_f_names(self.database, self.name, f_names_new)

    @classmethod
    def copy_descriptors_from_to(cls, database, from_name, to_name):
        if not cls.descriptors_compatible(database, from_name, to_name):
            message = f"Database {database}'s descriptors {from_name} and {to_name} are not compatible"
            raise ValueError(message)

        raise NotImplementedError

    @classmethod
    def descriptors_compatible(cls, database, first_name, second_name):
        from_des_type = CBIR(database, first_name).des_type
        to_des_type = CBIR(database, second_name).des_type
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
    def exists(cls, database, name):
        return (database in CBIR.get_databases()
                and name in CBIR.get_cbir_indexes_of_database(database)
                and cls._inited_properly(database, name))

    @classmethod
    def _inited_properly(cls, database, name):
        # TODO
        params_path = cls.get_params_path(database, name)
        return os.path.exists(params_path)

    def __str__(self):
        return f'CBIR {self.database} {self.name}'

    def empty(self):
        inverted_index = self.load_inverted_index()
        return len(inverted_index) == 0

    def load_fd(self):
        max_keypoints = self.max_keypoints

        if self.des_type == 'sift':
            fd = SIFT(max_keypoints=max_keypoints)
        elif self.des_type == 'surf':
            fd = SURF(max_keypoints=max_keypoints)
        elif self.des_type == 'l2net':
            fd = L2net_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif self.des_type == 'HardNetBrown':
            fd = HardNetBrown_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif self.des_type == 'HardNetAll':
            fd = HardNetAll_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        elif self.des_type == 'HardNetHPatches':
            fd = HardNetHPatches_des(max_keypoints=max_keypoints, use_cuda=CONFIG['use_cuda'])
        else:
            raise ValueError(f'Bad des_type: {self.des_type}')

        return fd

    def set_fd(self, fd):
        self.fd = fd

    def unset_fd(self):
        self.fd = None

    def load_ca(self):
        with open(CBIR.get_clusterer_path(self.database, self.name), 'rb') as f:
            ca = pickle.load(f)
        return ca

    def set_ca(self, ca):
        self.ca = ca

    def unset_ca(self):
        self.ca = None

    def get_candidates(self, query, filter=True):
        """
        :param query:  `[(visual_word, freq), ...]` query as a bow(BoVW)
        :param filter:
        """
        candidates = set()

        most_frequent = self.load_data_dependent_params()['most_frequent']

        interesting_words_in_query = [word
                                      for word, freq
                                      in enumerate(query)
                                      if (freq > 1e-7
                                          and (not filter or word not in most_frequent))]

        for word in interesting_words_in_query:
            new_candidates = self.get_candidates_by_word(word)
            candidates = candidates | new_candidates

        return candidates

    def get_candidates_by_word(self, word):
        inverted_index = self.load_inverted_index()
        return inverted_index[word]

    def search(self,
               img_path,
               n_candidates=100,
               topk=5, n_inliners_thr=20, max_verified=20,
               qe_avg=50, qe_limit=30, new_query=None,
               sv_enable=True, qe_enable=True, debug=False):
        """
        :param list_paths_to_images: query images
        :return: list paths images most similar to the query
        """
        start = time.time()
        # STEP 1. APPLY INVERTED INDEX TO GET CANDIDATES

        fd_loaded_before = self.fd is not None
        if not fd_loaded_before:
            self.set_fd(self.load_fd())

        ca_loaded_before = self.ca is not None
        if not ca_loaded_before:
            self.set_ca(self.load_ca())

        result_tuple = self.get_descriptor(img_path, both=True, total_count_coordinate_for_bow=True)

        if len(result_tuple) != 3 or result_tuple[0] is None:
            message = f'Could not get descriptor for image {img_path}'
            if not fd_loaded_before:
                self.unset_fd()
            raise ValueError(message)

        img_descriptor, img_bovw, kp = result_tuple
        print("Descriptor for query got in {}".format(time.time() - start))

        start = time.time()

        # if new_query is not None:
        #     img_bovw = new_query[:-1].astype(np.float32) / new_query[-1]
        # else:
        #     img_bovw = img_bovw[:-1].astype(np.float32) / img_bovw[-1]
        if new_query is not None:
            img_bovw = new_query.astype(np.float32)
        else:
            img_bovw = img_bovw.astype(np.float32)

        candidates = self.get_candidates(img_bovw[:-1])
        if len(candidates) == 0:
            candidates = self.get_candidates(img_bovw[:-1], filter=False)
        print("Candidates got in {}".format(time.time() - start))
        if debug:
            print(len(candidates))

        # STEP 2. PRELIMINARY RANKING
        start = time.time()

        bow = self.load_bow()
        idf = self.load_data_dependent_params()['idf']
        f_names = self.load_f_names()

        ranks = []
        for i in candidates:
            bow_row = np.array(bow[i].todense(), dtype=np.float).flatten()
            ranks.append((i, euclidean(img_bovw[:-1] / img_bovw[-1] * idf, bow_row[:-1] / bow_row[-1] * idf)))

        ranks = sorted(ranks, key=lambda x: x[1])
        candidates = [(i[0], f_names[i[0]]) for i in ranks[:n_candidates]]

        print("Short list got in {}".format(time.time() - start))
        if debug:
            pass

        # STEP 3. SPATIAL VERIFICATION
        if not sv_enable:
            # for compatibility with the output format
            candidates = [(c, 0) for c in candidates]
            if not fd_loaded_before:
                self.unset_fd()
            if not ca_loaded_before:
                self.unset_ca()
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
                bow_row = np.array(bow[r[0][0]].todense()).flatten()
                top_res.append(bow_row)
            one_more_query = (sum(top_res) + img_bovw) / (len(top_res) + 1)

            new_sv_candidates = self.search(img_path, n_candidates,
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

        if not fd_loaded_before:
            self.unset_fd()
        if not ca_loaded_before:
            self.unset_ca()
        return sv_candidates[:topk]

    def ransac(self, img_descriptor, kp, candidates,
               min_inliners, max_verified):

        index = self.load_index()
        descriptors = [index[img[1]] for img in candidates]

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

    # Very common
    @classmethod
    def get_storage_path(cls, database, name):
        return str(Path(cbir.DATABASES) / database / name)

    @classmethod
    def _save_params(cls, database, name,
                     params):
        with open(cls.get_params_path(database, name), 'wb') as f:
            pickle.dump(params, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_data_dependent_params(cls, database, name,
                                    data_dependent_params):
        with open(cls.get_data_dependent_params_path(database, name), 'wb') as f:
            pickle.dump(data_dependent_params, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_index(cls, database, name,
                    index):
        with open(cls.get_des_path(database, name), 'wb') as f:
            pickle.dump({k: (v[0],
                             [p.pt[0] for p in v[1]],
                             [p.pt[1] for p in v[1]],
                             [p.size for p in v[1]])
                         for k, v in index.items()}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_clusterer(cls, database, name,
                        clusterer):
        with open(cls.get_clusterer_path(database, name), 'wb') as f:
            pickle.dump(clusterer, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_bow(cls, database, name,
                  bow):
        with open(cls.get_bow_path(database, name), 'wb') as f:
            pickle.dump(bow, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_inverted_index(cls, database, name,
                             inverted_index):
        with open(cls.get_inverted_index_path(database, name), 'wb') as f:
            pickle.dump(inverted_index, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_f_names(cls, database, name,
                      f_names):
        with open(cls.get_f_names_path(database, name), 'wb') as f:
            pickle.dump(f_names, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _save_freqs(cls, database, name,
                    freqs):
        with open(cls.get_freqs_path(database, name), 'wb') as f:
            pickle.dump(freqs, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def get_params_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'params' + postfix)

    @classmethod
    def get_data_dependent_params_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'data_dependent_params' + postfix)

    @classmethod
    def get_des_path(cls, database, name):
        postfix_des = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'des' + postfix_des)

    @classmethod
    def get_clusterer_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'clusterer' + postfix)

    @classmethod
    def get_inverted_index_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'inverted_index' + postfix)

    @classmethod
    def get_bow_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'bow' + postfix)

    @classmethod
    def get_f_names_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'f_names' + postfix)

    @classmethod
    def get_freqs_path(cls, database, name):
        postfix = '.pkl'
        return os.path.join(cls.get_storage_path(database, name), 'freqs' + postfix)

    def load_params(self):
        with open(CBIR.get_params_path(self.database, self.name), 'rb') as f:
            params = pickle.load(f)
        return params

    def load_data_dependent_params(self):
        with open(CBIR.get_data_dependent_params_path(self.database, self.name), 'rb') as f:
            data_dependent_params = pickle.load(f)
        return data_dependent_params

    def load_inverted_index(self):
        with open(CBIR.get_inverted_index_path(self.database, self.name), 'rb') as f:
            inverted_index = pickle.load(f)
        return inverted_index

    def load_index(self):
        with open(CBIR.get_des_path(self.database, self.name), 'rb') as f:
            index = pickle.load(f)
        index = {k: (v[0],
                     [cv2.KeyPoint(*el) for el in zip(v[1], v[2], v[3])])
                 for k, v in index.items()}
        return index

    def load_f_names(self):
        with open(CBIR.get_f_names_path(self.database, self.name), 'rb') as f:
            f_names = pickle.load(f)
        return f_names

    def load_bow(self):
        with open(CBIR.get_bow_path(self.database, self.name), 'rb') as f:
            bow = pickle.load(f)
        return bow

    def load_freqs(self):
        with open(CBIR.get_freqs_path(self.database, self.name), 'rb') as f:
            freqs = pickle.load(f)
        return freqs


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
