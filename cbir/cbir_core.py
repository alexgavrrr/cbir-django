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
from cbir import database_service
from cbir.legacy_utils import draw_result
from cbir.models.learned_descriptors import HardNetAll_des, HardNetBrown_des, HardNetHPatches_des
from cbir.models.learned_descriptors import L2net_des
from cbir.models.learned_descriptors import SURF, SIFT
from cbir.vocabulary_tree import VocabularyTree

logger = logging.getLogger('cbir.photo_storage_inverted_file')

DES_TYPE = 'l2net'
MAX_KEYPOINTS = 2000


# TODO:
# Use DATABASES_RELATIVE_TO_BASE_DIR instead of DATABASES

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

        instance.db = database_service.get_db(cls.get_storage_path(database, name))

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
    def exists(cls, database, name):
        return (database in CBIR.get_databases()
                and name in CBIR.get_cbir_indexes_of_database(database)
                and cls._inited_properly(database, name))

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
        data_dependent_params['idf'] = np.zeros(params['n_words'], dtype=np.float32)
        data_dependent_params['freqs'] = np.zeros(params['n_words'], dtype=np.int32)
        data_dependent_params['most_frequent'] = []
        data_dependent_params['least_frequent'] = []

        cls._save_params(database, name, params)
        cls._save_data_dependent_params(database, name, data_dependent_params)

        db = database_service.get_db(cls.get_storage_path(database, name))
        database_service.create_empty(db)

        clusterer = None
        cls._save_clusterer(database, name, clusterer)

    @classmethod
    def _inited_properly(cls, database, name):
        params_path = cls.get_params_path(database, name)
        data_dependent_params_path = cls.get_data_dependent_params_path(database, name)

        potential_db = database_service.get_db(cls.get_storage_path(database, name))
        inited_properly = database_service.inited_properly(potential_db)

        return (os.path.exists(params_path)
                and os.path.exists(data_dependent_params_path)
                and inited_properly)

    def __str__(self):
        return f'CBIR {self.database} {self.name}'

    def empty(self):
        # TODO: Rewrite.
        inverted_index = self.load_inverted_index()
        return len(inverted_index) == 0

    def compute_descriptors(self,
                            list_paths_to_images,
                            to_index,
                            for_training_clusterer):
        """
        Computes decriptors for given images
        and saves these descriptors in search structures (sqlite).

        :param list_paths_to_images: images for which to compute descriptors
        :param to_index: whether these photos to be indexed
        :param for_training_clusterer: whether to use these photos for training clusterer
        """
        # TODO: fd_ready_decorator and ca_ready_decorator
        fd_loaded_before = self.fd is not None
        if not fd_loaded_before:
            self.set_fd(self.load_fd())

        count_new = 0
        count_defects = 0
        count_old = 0
        for path_to_image in tqdm(list_paths_to_images, desc='Computing descriptors for photos and saving in the database'):
            if database_service.is_image_indexed(self.db, path_to_image):
                count_old += 1
            else:
                descriptor_now = self.get_descriptor(path_to_image, raw=True)
                if descriptor_now[0] is not None:
                    count_new += 1
                    new_photo = {
                        'name': path_to_image,
                        'descriptor': self.serialize_descriptor(descriptor_now),
                        'to_index': to_index,
                        'for_training': for_training_clusterer
                    }
                    database_service.add_photos(self.db, [new_photo])
                else:
                    count_defects += 1
                    print("No keypoints found for {}".format(path_to_image))

        if not fd_loaded_before:
            self.unset_fd()

        print(f'count_new: {count_new} ; count_defects: {count_defects} ; count_old: {count_old}')

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

    def train_clusterer(self):
        """
        Trains clusterer on images which's descriptors
        have already been computed and marked for_training.
        """
        logger.info(f'Training clusterer on previously computed descriptors for index {self.name} of database {self.database}')
        SIZE_DESCRIPTOR = 128
        COUNT_DESCRIPTORS_EXPECTED = 5000
        PLACEHOLDER_SIZE = 1000

        path_to_mmap_descriptors = Path(CBIR.get_storage_path(self.database, self.name)) / 'mmap_descriptors'
        mmap_descriptos = np.memmap(path_to_mmap_descriptors,
                                    dtype='float32',
                                    shape=(COUNT_DESCRIPTORS_EXPECTED, SIZE_DESCRIPTOR),
                                    mode='w+')
        placeholder = np.zeros(shape=(PLACEHOLDER_SIZE, SIZE_DESCRIPTOR), dtype='float32')
        begin_placehoder = 0
        begin_mmap_descriptors = 0
        logger.info('Saving descriptors from database to disk')

        count_photos_for_training = 0
        count_descriptors_flattened_for_training = 0
        for descriptor in tqdm(database_service.get_photos_descriptors_for_training_iterator(self.db),
                               desc="Saving descriptors from database to disk"):
            descriptor = descriptor['descriptor']
            deserialized_descriptor = self.deserialize_descriptor(descriptor)
            deserialized_descriptor_without_kp = deserialized_descriptor[0]
            end_now = min(begin_placehoder + deserialized_descriptor_without_kp.shape[0], placeholder.shape[0])

            count_photos_for_training += 1
            count_descriptors_flattened_for_training += deserialized_descriptor_without_kp.shape[0]

            # TODO: Remove comment
            # print(f'deserialized_descriptor.shape: {deserialized_descriptor.shape}')
            # print(f'end_now: {end_now}')
            # print(f'begin_placehoder: {begin_placehoder}')
            # print(f'end_now - begin_placehoder: {end_now - begin_placehoder}')

            placeholder[begin_placehoder:end_now] = deserialized_descriptor_without_kp[0: end_now - begin_placehoder]
            begin_placehoder = end_now
            if end_now == placeholder.shape[0]:
                mmap_descriptos[begin_mmap_descriptors: begin_mmap_descriptors + placeholder.shape[0]] = placeholder
                begin_placehoder = 0
                begin_mmap_descriptors += placeholder.shape[0]
        mmap_descriptos[begin_mmap_descriptors: begin_mmap_descriptors + end_now] = placeholder[:end_now]

        logger.info(f'Count descriptors flattened for training: {count_descriptors_flattened_for_training}\n'
                    f'Count photos used for training: {count_photos_for_training}\n'
                    f'Average count of descriptors - keypoints on one photo: '
                    f'{int(count_descriptors_flattened_for_training / count_photos_for_training)})')

        def loader(indices):
            if len(indices) > placeholder.shape[0]:
                raise ValueError
            placeholder[:len(indices)] = mmap_descriptos[indices]
            return placeholder

        ca = VocabularyTree(L=self.L, K=self.K).fit(loader)
        CBIR._save_clusterer(self.database, self.name, ca)

    def add_images_to_index(self):
        """
        Adds images marked for indexing,
        whose descriptors have already been computed, to search index.
        If images have already been indexed before (bow is not null) then ignores it.
        """
        logger.info(f'Adding photos to index {self.name} in database {self.database}')
        # TODO: fd_ready_decorator and ca_ready_decorator
        ca_loaded_before = self.ca is not None
        if not ca_loaded_before:
            self.set_ca(self.load_ca())

        # TODO: Apply cunning trick. Model WordPhoto should not have index by word in the beginning
        # for faster inserts. When all inserts are done then we should build index and get blobs sorted by word.

        data_dependent_params = self.load_data_dependent_params()
        freqs = data_dependent_params['freqs']
        for photo in tqdm(database_service
                                  .get_photos_descriptors_needed_to_add_to_index_iterator(self.db),
                          desc='Applying clusterer, updaing bow and inverted index '
                               'for every photo to needed to add to index'):
            word_photo_relations = []
            photo_descriptor = self.deserialize_descriptor(photo.descriptor)
            photo_words = self.ca.predict(photo_descriptor[0])
            photo_bow = np.zeros((self.n_words + 1,), dtype=np.int16)
            for word in photo_words:
                word_photo_relations += [{'word': word, 'photo': photo.name}]
                photo_bow[word] += 1
                photo_bow[self.n_words] += 1
                if photo_bow[word] == 1:
                    freqs[word] += 1

            # TODO: photo.pk instead of photo.name must be in the future.
            database_service.add_word_photo_relations(self.db, word_photo_relations)
            photo_to_update = {
                'name': photo.name,
                'bow': self.serialize_bow(photo_bow)
            }
            database_service.update_bows(self.db, [photo_to_update])

        # TODO: Build index on WordPhoto by word if not yet.

        # Sort by word and get word photo relations
        word_now = None
        word_photos_now = set()
        for ind, word_photo_relation in enumerate(database_service.get_word_photo_relations_sorted(self.db)):
            word_next = word_photo_relation['word']
            photo_next = word_photo_relation['photo']
            if word_now and word_now != word_next:
                word_to_insert = {
                    'word': word_now,
                    'photos': self.serialize_word_photos(word_photos_now)
                }
                database_service.insert_or_replace_word(self.db, [word_to_insert])
                word_photos_now = set()

            word_now = word_next
            word_photos_now.add(photo_next)

        if word_now:
            word_to_insert = {
                'word': word_now,
                'photos': self.serialize_word_photos(word_photos_now)
            }
            database_service.insert_or_replace_word(self.db, [word_to_insert])

        five_percent = int(0.085 * self.n_words)
        freqs = np.argsort(freqs)
        most_frequent = freqs[-five_percent:]
        least_frequent = freqs[:five_percent]

        total_count_photos_indexed = database_service.count_indexed(self.db)
        idf = compute_idf_lazy(freqs, total_count_photos_indexed)

        data_dependent_params = {}
        data_dependent_params['freqs'] = freqs
        data_dependent_params['idf'] = idf
        data_dependent_params['most_frequent'] = most_frequent
        data_dependent_params['least_frequent'] = least_frequent

        CBIR._save_data_dependent_params(self.database, self.name, data_dependent_params)

    def get_candidates_raw(self, query, filter=True):
        """
        :param query:  `[(visual_word, freq), ...]` query as a bow(BoVW)
        :param filter:
        """
        most_frequent = self.load_data_dependent_params()['most_frequent']

        interesting_words_in_query = [word
                                      for word, freq
                                      in enumerate(query)
                                      if (freq > 1e-7
                                          and (not filter or word not in most_frequent))]

        candidates_iterator = database_service.get_photos_by_words_iterator(self.db, interesting_words_in_query)
        candidates_iterator_modified = (candidate['photos'] for candidate in candidates_iterator)
        return list(candidates_iterator_modified)

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
        logger.info(f'Performing search in database {self.database} on index {self.name}. '
                    f'Path to query: {img_path}')
        start = time.time()
        # STEP 1. APPLY INVERTED INDEX TO GET CANDIDATES

        # TODO: fd_ready_decorator and ca_ready_decorator
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

        if new_query is not None:
            img_bovw = new_query.astype(np.float32)
        else:
            img_bovw = img_bovw.astype(np.float32)

        candidates_raw = self.get_candidates_raw(img_bovw[:-1])
        if len(candidates_raw) == 0:
            candidates_raw = self.get_candidates_raw(img_bovw[:-1], filter=False)
        candidates = set()
        for candidate_raw in candidates_raw:
            candidates |= self.deserialize_word_photos(candidate_raw)
        print("Candidates got in {}".format(time.time() - start))
        if debug:
            print(len(candidates))

        # TODO: In the future make candidates an iterator. Now Algorithm requires all candidates in RAM
        # because it applies sorting. But algorithm needs only top n_candidates.
        # Thus we can apply linear algorithm with heapq or tree and iterator will be enough.

        print(f'type(next(iter(candidates))): {type(next(iter(candidates)))}')
        print(f'next(iter(candidates)): {next(iter(candidates))}')

        # STEP 2. PRELIMINARY RANKING
        start = time.time()

        # bow = self.load_bow()
        idf = self.load_data_dependent_params()['idf']

        # f_names = self.load_f_names()

        # TODO: Use heapq to obtain preliminary top n_candidates
        # Getting all candidates in ram and sorting can infeasible.
        ranks = []
        for candidate in candidates:
            candidate_bow_raw = database_service.get_bow(self.db, candidate)
            candidate_bow_raw = candidate_bow_raw['bow']
            candidate_bow = self.deserialize_bow(candidate_bow_raw)
            # bow_row = np.array(bow[candidate].todense(), dtype=np.float).flatten()
            ranks.append((candidate, euclidean(img_bovw[:-1] / img_bovw[-1] * idf,
                                               candidate_bow[:-1] / candidate_bow[-1] * idf)))

        ranks = sorted(ranks, key=lambda x: x[1])

        print(f'Ranks: {ranks}')

        # candidates = [(i[0], f_names[i[0]]) for i in ranks[:n_candidates]]
        # earlier [(ind_candidate, name_candidate)]

        candidates = [(None, rank[0]) for rank in ranks[:n_candidates]]

        print(f'Candidates: {candidates}')

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

        # TODO: Do not return and get descriptors_kp if it is not needed.
        [all_matches, all_matches_masks, all_transforms,
         verified,
         descriptors_kp] = self.ransac(img_descriptor, kp, candidates,
                                       n_inliners_thr, max_verified)

        print(f'all_matches: {type(all_matches)}')
        print(f'all_matches[0]: {all_matches[0]}')
        print(f'len(all_matches): {len(all_matches)}')

        print(f'all_matches_masks: {type(all_matches_masks)}')
        print(f'all_matches_masks[0]: {all_matches_masks[0]}')
        print(f'len(all_matches_masks): {len(all_matches_masks)}')

        print(f'all_transforms: {type(all_transforms)}')
        print(f'all_transforms[0]: {all_transforms[0]}')
        print(f'len(all_transforms): {len(all_transforms)}')

        print(f'verified: {type(verified)}')
        print(f'verified[0]: {verified[0]}')
        print(f'len(verified): {len(verified)}')

        print(f'descriptors_kp: {type(descriptors_kp)}')
        print(f'descriptors_kp[0]: {descriptors_kp[0]}')
        print(f'len(descriptors_kp): {len(descriptors_kp)}')

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
            for sv_candidate in sv_candidates[:qe_avg]:
                # INDEX CANDIDATE NEEDED HERE
                # bow_row = np.array(bow[r[0][0]].todense()).flatten()

                sv_candidate_bow_raw = database_service.get_bow(self.db, sv_candidate[0][1])
                sv_candidate_bow_raw = sv_candidate_bow_raw['bow']
                sv_candidate_bow = self.deserialize_bow(sv_candidate_bow_raw)
                top_res.append(sv_candidate_bow)

            one_more_query = (sum(top_res) + img_bovw) / (len(top_res) + 1)

            new_sv_candidates = self.search(img_path, n_candidates,
                                            topk, n_inliners_thr, max_verified,
                                            qe_avg, qe_limit, one_more_query,
                                            qe_enable=qe_enable, sv_enable=sv_enable,
                                            debug=debug)

            old = set(sv_candidates[i][0][0] for i in range(len(sv_candidates)))
            new = set(new_sv_candidates[i][0][0] for i in range(len(new_sv_candidates)))
            duplicates = set(old) & set(new)

            new_sv_candidates = [el for el in new_sv_candidates
                                 if el[0][0] not in duplicates]

            # CANDIDATES' RANKS VALUES ARE NEEDED HERE.
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

        # index = self.load_index()
        # descriptors = [index[img[1]] for img in candidates]
        # descriptors, descriptors_kp = list(zip(*descriptors))

        descriptors_raw_iterator = database_service.get_photos_descriptors_by_names_iterator(
            self.db,
            # TODO: `candidate[1]` because not it is this way for backward-compatibility.
            # In the future I will choose ind or name as the only indentifier of a photo
            [candidate[1]
             for candidate
             in candidates]
        )
        descriptors_kp_pair_iterator = (self.deserialize_descriptor(descriptor_raw['descriptor'])
                                        for descriptor_raw in descriptors_raw_iterator)

        # Brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2)

        MIN_MATCH_COUNT = 10
        all_matches = []
        all_matches_masks = []
        all_transforms = []

        verified = []
        n_verified = 0

        # TODO: Now it is for backward-compatibility. Remove in the future
        descriptors_kp = []

        for i, descriptor_kp_pair in enumerate(descriptors_kp_pair_iterator):
            descriptor = descriptor_kp_pair[0]
            descriptor_kp = descriptor_kp_pair[1]
            descriptors_kp += [kp]

            matches = bf.match(img_descriptor, descriptor)
            if len(matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([descriptor_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

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
        with open(CBIR.get_data_dependent_params_path(self.database, self.name), 'rb') as f:
            freqs = pickle.load(f)['freqs']
        return freqs

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

    def serialize_descriptor(self, descriptor):
        return pickle.dumps((descriptor[0],
                             [p.pt[0] for p in descriptor[1]],
                             [p.pt[1] for p in descriptor[1]],
                             [p.size for p in descriptor[1]]),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_descriptor(self, descriptor):
        descriptor = pickle.loads(descriptor)
        return (descriptor[0], [cv2.KeyPoint(*el) for el in zip(descriptor[1], descriptor[2], descriptor[3])])

    def serialize_bow(self, bow):
        bow_sparse = sparse.coo_matrix(bow)
        return pickle.dumps(bow_sparse)

    def deserialize_bow(self, bow):
        bow_sparse = pickle.loads(bow)
        bow = bow_sparse.toarray().squeeze()
        return bow

    def serialize_word_photos(self, word_photos):
        word_photos = pickle.dumps(word_photos)
        return word_photos

    def deserialize_word_photos(self, word_photos):
        return pickle.loads(word_photos)


def compute_idf_lazy(freqs, total_count_documents):
    not_zero = np.where(freqs != 0)[0]
    idf = np.zeros(freqs.shape[0])
    idf[not_zero] = np.log(total_count_documents / freqs[not_zero])
    return idf


def main(storage_path, training_path, label):
    # TODO: Write or delete

    test_path = storage_path
    cbir_index = CBIR.get_instance()

    while True:
        print("Enter command")

        break

    print("Finished")


if __name__ == "__main__":
    main(
        '/Users/alexgavr/main/Developer/Data/Buildings/Revisited/datasets/roxford5k_sample/jpg',
        '/Users/alexgavr/main/Developer/Data/Buildings/Revisited/datasets/roxford5k_sample/jpg',
        '')