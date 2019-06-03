import functools
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
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import cbir
from cbir import CONFIG
from cbir import database_service
from cbir.models.learned_descriptors import HardNetAll_des, HardNetBrown_des, HardNetHPatches_des
from cbir.models.learned_descriptors import L2net_des
from cbir.models.learned_descriptors import SURF, SIFT
from cbir.vocabulary_tree import VocabularyTree

logger = logging.getLogger('cbir.cbir_core')


class CBIRCore:
    # staticmethod. Look bottom of the class
    def decorator_load_fd_if_needed(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            fd_loaded_before = self.fd is not None
            if not fd_loaded_before:
                self.set_fd(self.load_fd())
            result = func(self, *args, **kwargs)
            if not fd_loaded_before:
                self.unset_fd()
            return result

        return wrap

    # staticmethod. Look bottom of the class
    def decorator_load_ca_if_needed(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            ca_loaded_before = self.ca is not None
            if not ca_loaded_before:
                self.set_ca(self.load_ca())
            result = func(self, *args, **kwargs)
            if not ca_loaded_before:
                self.unset_ca()
            return result

        return wrap

    # staticmethod. Look bottom of the class
    def decorator_load_bow_if_needed(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            bow_loaded_before = self.bow is not None
            if not bow_loaded_before:
                self.set_bow(self.load_bow())
            result = func(self, *args, **kwargs)
            if not bow_loaded_before:
                self.unset_bow()
            return result

        return wrap

    # staticmethod. Look bottom of the class
    def decorator_load_inv_if_needed(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            inv_loaded_before = self.inv is not None
            if not inv_loaded_before:
                self.set_inv(self.load_inv())
            result = func(self, *args, **kwargs)
            if not inv_loaded_before:
                self.unset_inv()
            return result

        return wrap

    # staticmethod. Look bottom of the class
    def decorator_load_most_frequent_if_needed(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            most_frequent_loaded_before = self.most_frequent is not None
            if not most_frequent_loaded_before:
                self.set_most_frequent(self.load_most_frequent())
            result = func(self, *args, **kwargs)
            if not most_frequent_loaded_before:
                self.unset_most_frequent()
            return result

        return wrap


    @classmethod
    def get_instance(cls, database, name, dummy=False):
        instance = CBIRCore()
        instance.database = database
        instance.name = name

        if dummy:
            return instance

        if not cls.exists(database, name):
            message = (f'CBIRCore {name} in database {database} does not exist. First create it. For example, by calling '
                       f'`create_empty`.')
            raise ValueError(message)

        params = instance.load_params()
        instance.des_type = params['des_type']
        instance.max_keypoints = params['max_keypoints']
        instance.K = params['K']
        instance.L = params['L']
        instance.n_words = instance.K ** instance.L

        instance.db = database_service.get_db(cls.get_storage_path(database, name))

        instance.fd = None
        instance.ca = None
        instance.bow = None
        instance.inv = None
        instance.most_frequent = None

        return instance

    def get_indexed_photos(self, limit=None):
        photos = database_service.get_indexed_photos(self.db, limit)
        return photos


    @classmethod
    def prepare_place_for_database_if_needed(cls, database):
        if not os.path.exists(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database):
            os.mkdir(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database)

    @classmethod
    def prepare_place_for_cbir_index_if_needed(cls, database, name):
        if not os.path.exists(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database / name):
            os.mkdir(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database / name)

    @classmethod
    def create_empty_if_needed(cls, database, name,
                               des_type, max_keypoints,
                               K, L):
        """
        Creates empty CBIRCore instance which indexes 0 objects.
        :param database:
        :param name:
        :param des_type:
        :param max_keypoints:
        :param K:
        :param L:
        :return: whether created_now
        """
        if not cls.exists(database, name):
            cls.prepare_place_for_database_if_needed(database)
            cls.prepare_place_for_cbir_index_if_needed(database, name)
            cls._init_search_structures(database, name,
                                        des_type, max_keypoints, K, L)
            return True
        return False


    @classmethod
    def exists(cls, database, name):
        return (database in CBIRCore.get_databases()
                and name in CBIRCore.get_cbir_indexes_of_database(database)
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
        logger.info(f'Initing cbir index with params: {params}')

        data_dependent_params = {}
        data_dependent_params['count_images'] = 1  # 0th image is empty. It is needed as rowid begins from 1.
        data_dependent_params['idf'] = np.zeros(params['n_words'], dtype=np.float32)
        data_dependent_params['freqs'] = np.zeros(params['n_words'], dtype=np.int32)
        data_dependent_params['most_frequent'] = set()
        data_dependent_params['least_frequent'] = set()

        cls._save_params(database, name, params)
        cls._save_data_dependent_params(database, name, data_dependent_params)

        db = database_service.get_db(cls.get_storage_path(database, name))
        database_service.create_empty(db)

        clusterer = []

        # 0th empty row in bow because rowid in sqlite table begins from 0.
        bow = sparse.csr_matrix([], shape=(1, params['n_words'] + 1))

        inverted_index = [set() for i in range(params['n_words'])]
        freqs = np.zeros(params['n_words'], dtype=np.int16)

        cls._save_params(database, name, params)
        cls._save_data_dependent_params(database, name, data_dependent_params)
        cls._save_clusterer(database, name, clusterer)
        cls._save_bow(database, name, bow)
        cls._save_inverted_index(database, name, inverted_index)
        cls._save_freqs(database, name, freqs)

    def set_K_L(self, K, L):
        logger.info(f'Setting K, L: {K}, {L}')
        params = self.load_params()
        params['K'] = K
        params['L'] = L
        params['n_words'] = K ** L
        CBIRCore._save_params(self.database, self.name, params)

        data_dependent_params = self.load_data_dependent_params()
        data_dependent_params['idf'] = np.zeros(params['n_words'], dtype=np.float32)
        data_dependent_params['freqs'] = np.zeros(params['n_words'], dtype=np.int32)
        data_dependent_params['most_frequent'] = set()
        data_dependent_params['least_frequent'] = set()
        CBIRCore._save_data_dependent_params(self.database, self.name, data_dependent_params)

        clusterer = []
        # 0th empty row in bow because rowid in sqlite table begins from 0.
        bow = sparse.csr_matrix([], shape=(1, params['n_words'] + 1))
        inverted_index = [set() for i in range(params['n_words'])]
        freqs = np.zeros(params['n_words'], dtype=np.int16)
        CBIRCore._save_clusterer(self.database, self.name, clusterer)
        CBIRCore._save_bow(self.database, self.name, bow)
        CBIRCore._save_inverted_index(self.database, self.name, inverted_index)
        CBIRCore._save_freqs(self.database, self.name, freqs)

    @classmethod
    def _inited_properly(cls, database, name):
        params_path = cls.get_params_path(database, name)
        data_dependent_params_path = cls.get_data_dependent_params_path(database, name)

        if not os.path.exists(params_path):
            return False
        if not os.path.exists(data_dependent_params_path):
            return False

        potential_db = database_service.get_db(cls.get_storage_path(database, name))
        database_inited_properly = database_service.inited_properly(potential_db, external_mode=False)

        dummy_cbir_core = CBIRCore.get_instance(database, name, dummy=True)
        search_structures_inited_properly = True
        params = dummy_cbir_core.load_params()
        n_words = params['n_words']
        bow = dummy_cbir_core.load_bow()
        if not isinstance(bow, sparse.csr_matrix):
            search_structures_inited_properly = False
        elif bow.shape[1] != n_words + 1:
            search_structures_inited_properly = False

        return (os.path.exists(params_path)
                and os.path.exists(data_dependent_params_path)
                and database_inited_properly
                and search_structures_inited_properly)

    def __str__(self):
        return f'CBIRCore {self.database} {self.name}'

    @decorator_load_fd_if_needed
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
        count_new = 0
        count_defects = 0
        count_old = 0
        BUFFER_CAPACITY = 1000
        buffered_new_photos = []
        start = time.time()
        for path_to_image in tqdm(list_paths_to_images, desc='Computing descriptors for photos and saving in the database'):
            if database_service.is_image_descriptor_computed(self.db, path_to_image):
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
                    buffered_new_photos += [new_photo]
                    if len(buffered_new_photos) == BUFFER_CAPACITY:
                        database_service.add_photos_descriptors(self.db, buffered_new_photos)
                        buffered_new_photos = []
                else:
                    count_defects += 1
                    logger.debug("No keypoints found for {}".format(path_to_image))

        if buffered_new_photos:
            database_service.add_photos_descriptors(self.db, buffered_new_photos)
            buffered_new_photos = []

        time_computing_and_writing_descriptors = round(time.time() - start, 3)
        average_time_computing_and_writing_descriptors = None if not count_new else round(time_computing_and_writing_descriptors / count_new, 3)

        logger.info(f'count_new: {count_new} ; count_defects: {count_defects} ; count_old: {count_old}')
        logging.getLogger('profile.computing_descriptors').info(
            f'time_computing_and_writing_descriptors={time_computing_and_writing_descriptors},'
            f'average_time_computing_and_writing_descriptors={average_time_computing_and_writing_descriptors},'
        )

    def get_descriptor(self, path_to_image, raw=False, both=False, total_count_coordinate_for_bow=True):
        if type(path_to_image) is str:
            image = cv2.imread(path_to_image, 0)
        elif type(path_to_image) is np.ndarray:
            image = path_to_image
        else:
            logger.debug(f"Unknown type: {type(path_to_image)}")
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

    def get_bow_vector(self, img_des, total_count=True, csr_matrix_format=False):
        if img_des.ndim == 1:
            img_des = img_des.reshape(img_des.shape[0], -1)

        if csr_matrix_format:
            col_indices = np.empty(
                shape=(img_des.shape[0] + (1 if total_count else 0),),
                dtype='uint32')
            col_indices[:img_des.shape[0]] = self.ca.predict(img_des)
            if total_count:
                col_indices[img_des.shape[0]] = self.n_words

            values = np.ones(
                (img_des.shape[0] + (1 if total_count else 0),),
                dtype='uint16')
            if total_count:
                values[img_des.shape[0]] = img_des.shape[0]

            # uint 16 is enough for max_kp < 64000
            return sparse.csr_matrix(
                (values, col_indices, [0, len(col_indices)]),
                shape=(1, self.n_words + 1),
                dtype='uint16')
        else:
            res = np.zeros(
                shape=(self.n_words + (1 if total_count else 0),),
                dtype='uint16')
            pred = self.ca.predict(img_des)
            for p in pred:
                res[p] += 1
            if total_count:
                res[self.n_words] = len(pred)
            return res

    def train_clusterer(self):
        """
        Trains clusterer on images which's descriptors
        have already been computed and marked for_training.
        """
        logger.info(f'Training clusterer for index {self.name} of database {self.database}')

        # count_photos_for_training_expected = database_service.count_for_training(self.db)
        # COUNT_DESCRIPTORS_EXPECTED = self.max_keypoints * count_photos_for_training_expected

        # sift1b_pqcodes_path = 'sift1b_pqcodes.npy'
        # sift1b_encoder_path = 'encoder_1.pkl'
        sift1b_pqcodes_path = None
        sift1b_encoder_path = None
        if sift1b_encoder_path:
            with open(sift1b_encoder_path, 'rb') as fin:
                sift1b_encoder = pickle.load(fin)
        else:
            sift1b_encoder = None

        def data_loader():
            for descriptor in database_service.get_photos_descriptors_for_training_iterator(self.db):
                yield from self.deserialize_descriptor(descriptor['descriptor'])[0]

        start = time.time()
        ca = VocabularyTree(L=self.L, K=self.K).fit(
            data_loader(),
            sift1b_encoder=sift1b_encoder,
            sift1b_pqcodes_path=sift1b_pqcodes_path,
            format='uint8' if self.des_type == 'sift' else 'float32')
        time_fitting_vocabulary_tree = round(time.time() - start, 3)

        start = time.time()
        CBIRCore._save_clusterer(self.database, self.name, ca)
        time_saving_vocabulary_tree = round(time.time() - start, 3)

        logging.getLogger('profile.training_clusterer').info(
            # f'time_copying_descriptors_to_memmap={time_copying_descriptors_to_memmap},'
            f'time_fitting_vocabulary_tree={time_fitting_vocabulary_tree},'
            f'time_saving_vocabulary_tree={time_saving_vocabulary_tree},'
        )

    @decorator_load_ca_if_needed
    def add_images_to_index(self):
        """
        Adds images marked for indexing,
        whose descriptors have already been computed, to search index.
        If images have already been indexed before (bow is not null) then ignores it.
        """
        logger.info(f'Adding photos to index {self.name} in database {self.database}')

        profile_add_images_to_index_logger = logging.getLogger('profile.add_images_to_index')

        data_dependent_params = self.load_data_dependent_params()
        before_count_photos_indexed = data_dependent_params['count_images']
        resulting_count_photos_indexed = database_service.count_for_indexing(self.db) + 1  # NOTE: `+ 1` as there is 0th empty obj because rowid starts from 1
        new_count_photos_to_index = resulting_count_photos_indexed - before_count_photos_indexed
        logger.info(f'before_count_photos_indexed: {before_count_photos_indexed}; resulting_count_photos_indexed: {resulting_count_photos_indexed}; new_count_photos_to_index: {new_count_photos_to_index}')

        inverted_index_new = [set() for _ in range(self.n_words)]

        count_values_expected = resulting_count_photos_indexed * self.max_keypoints

        # if count_photos < 4 * 10 ** 9 then uint32 is enough
        row_indices = np.empty(shape=(count_values_expected + resulting_count_photos_indexed,), dtype='uint32')

        # if voc_size < 4 * 10 ** 9 then uint32 is enough
        col_indices = np.empty(shape=(count_values_expected + resulting_count_photos_indexed,),
                                   dtype='uint32')

        # if max_kp < 64000 then uint16 is enough
        values = np.empty(shape=(count_values_expected + resulting_count_photos_indexed,),
                          dtype='uint16')
        len_coo_values = 0

        start = time.time()
        for photo in tqdm(database_service
                                  .get_photos_descriptors_needed_to_add_to_index_iterator(self.db, from_id=before_count_photos_indexed),
                          desc='Applying clusterer, updaing bow and inverted index '
                               'for every photo needed to add to index'):
            photo_descriptor = self.deserialize_descriptor(photo.descriptor)
            photo_words = self.ca.predict(photo_descriptor[0])

            values[len_coo_values: len_coo_values + photo_words.shape[0]] = 1
            values[len_coo_values + photo_words.shape[0]] = photo_words.shape[0]

            assert(photo.rowid >= before_count_photos_indexed)
            assert(photo.rowid < resulting_count_photos_indexed)

            # `photo.rowid - before_count_photos_indexed` because we will do vstack below
            row_indices[len_coo_values: len_coo_values + photo_words.shape[0]] = photo.rowid - before_count_photos_indexed
            row_indices[len_coo_values + photo_words.shape[0]] = photo.rowid - before_count_photos_indexed

            col_indices[len_coo_values: len_coo_values + photo_words.shape[0]] = photo_words
            col_indices[len_coo_values + photo_words.shape[0]] = self.n_words

            len_coo_values += photo_words.shape[0] + 1

            for word in photo_words:
                inverted_index_new[word].add(photo.rowid)

        row_indices.resize(len_coo_values)
        col_indices.resize(len_coo_values)
        values.resize(len_coo_values)
        time_creating_bow_and_inv = round(time.time() - start, 3)

        start = time.time()
        bow_old = self.load_bow()
        time_loading_old_bow = round(time.time() - start, 3)

        start = time.time()
        bow_new = sparse.vstack(
            (bow_old,
            sparse.coo_matrix(
                (values, (row_indices, col_indices)),
                shape=(new_count_photos_to_index, self.n_words + 1))
             ),
            format='csr'
        )
        time_vstack = round(time.time() - start, 3)

        start = time.time()
        CBIRCore._save_bow(self.database, self.name, bow_new)
        time_storing_new_bow = round(time.time() - start, 3)
        del row_indices, col_indices, values, bow_old, bow_new

        start = time.time()
        inverted_index_old = self.load_inv()
        time_loading_old_inv = round(time.time() - start, 3)

        start = time.time()
        for word in range(len(inverted_index_new)):
            inverted_index_new[word] |= inverted_index_old[word]
        time_combining_invs = round(time.time() - start, 3)

        freqs = data_dependent_params['freqs']
        for word in range(len(inverted_index_new)):
            freqs[word] = len(inverted_index_new[word])

        start = time.time()
        CBIRCore._save_inverted_index(self.database, self.name, inverted_index_new)
        time_storing_new_inv = round(time.time() - start, 3)
        del inverted_index_new, inverted_index_old


        start = time.time()
        little_percent = int(0.085 * self.n_words)

        #  Костыль нужный для тестовых ситуаций, когда little_percent = 0
        if little_percent == 0:
            little_percent = 1

        words_sorted_by_freqs = np.argsort(freqs)
        most_frequent = set(words_sorted_by_freqs[-little_percent:])
        least_frequent = set(words_sorted_by_freqs[:little_percent])
        time_sorting_and_finding_most_frequent_words = round(time.time() - start, 3)

        idf = compute_idf_lazy(freqs, resulting_count_photos_indexed)

        start = time.time()
        data_dependent_params = {}
        data_dependent_params['count_images'] = resulting_count_photos_indexed
        data_dependent_params['freqs'] = freqs
        data_dependent_params['idf'] = idf
        data_dependent_params['most_frequent'] = most_frequent
        data_dependent_params['least_frequent'] = least_frequent
        CBIRCore._save_data_dependent_params(self.database, self.name, data_dependent_params)
        time_storing_data_dependent_params = round(time.time() - start, 3)
        profile_add_images_to_index_logger.info(
            f'time_creating_bow_and_inv={time_creating_bow_and_inv},'
            f'time_loading_old_bow={time_loading_old_bow},'
            f'time_vstack={time_vstack},'
            f'time_storing_new_bow={time_storing_new_bow},'
            f'time_loading_old_inv={time_loading_old_inv},'
            f'time_combining_invs={time_combining_invs},'
            f'time_storing_new_inv={time_storing_new_inv},'
            f'time_sorting_and_finding_most_frequent_words={time_sorting_and_finding_most_frequent_words},'
            f'time_storing_data_dependent_params={time_storing_data_dependent_params},'
        )

    def get_candidates(self, query, filter=True, bad_words=[]):
        """
        :param query:  `[(visual_word, freq), ...]` query as a bow(BoVW)
        :param filter:
        """
        interesting_words_in_query = [word
                                      for word, freq
                                      in enumerate(query)
                                      if (freq > 1e-7
                                          and (not filter or (word not in self.most_frequent
                                                              and word not in bad_words)))]
        candidates = set()
        for word in interesting_words_in_query:
            candidates |= self.inv[word]
        return list(candidates)

    def log_answers(self, query, result, sv, qe):
        answers_logger = logging.getLogger('profile.answers')
        for rank, row in enumerate(result):
            answers_logger.info(f'{query},{row[0][1]},{rank + 1},{row[1]},{sv},{qe}')

    def log_search_times(self, times):
        search_times_logger = logging.getLogger('profile.search')
        string_parts = []
        for time_now in times:
            string_parts += [f'{time_now[0]}={time_now[1]}']
        search_times_logger.info(','.join(string_parts))

    def put_photo_names_to_result_return_time(self, result):
        start = time.time()
        for i in range(len(result)):
            rowid = result[i][0][0]
            [(_, name_now)] = database_service.get_names_by_rowids(self.db, [rowid])
            ((rowid, _), val) = result[i]
            result[i] = ((int(rowid), name_now), val)
        time_getting_names = round(time.time() - start, 3)
        return time_getting_names

    @decorator_load_fd_if_needed
    @decorator_load_ca_if_needed
    @decorator_load_bow_if_needed
    @decorator_load_inv_if_needed
    @decorator_load_most_frequent_if_needed
    def search(self,
               img_path,
               n_candidates=100,
               topk=5, n_inliners_thr=20, max_verified=50,
               qe_avg=50, qe_limit=30, new_query=None,
               sv_enable=True, qe_enable=True, debug=False,
               similarity_threshold=None,
               precomputed_img_descriptor=None, precomputed_kp=None,
               query_name=None,
               p_fine_max=None):
        """
        :param list_paths_to_images: query images
        :return: list paths images most similar to the query
        """
        p_fine_max = p_fine_max if p_fine_max is not None else 1.0
        query_name = query_name or img_path
        logger.info(f'Performing search in database {self.database} on index {self.name}. '
                    f'Query_name: {query_name}')
        logger.info(f'Search params: topk: {topk}, '
                     f'n_candidates: {n_candidates}, max_verified: {max_verified}, '
                     f'similarity_threshold: {similarity_threshold}, '
                     f'sv_enable: {sv_enable}, qe_enable: {qe_enable}, '
                     f'p_fine_max: {p_fine_max}')

        times = []  # [(time_name, time_value), ...]

        # STEP 1. APPLY INVERTED INDEX TO GET CANDIDATES

        start = time.time()
        data_dependent_params = self.load_data_dependent_params()
        time_loading_dependent_params = round(time.time() - start, 3)
        times +=[('time_loading_dependent_params', time_loading_dependent_params)]

        idf = data_dependent_params['idf']
        freqs = data_dependent_params['freqs']

        time_getting_descriptor = None
        if new_query is not None:
            img_descriptor = precomputed_img_descriptor
            kp = precomputed_kp
            img_bovw = new_query

            # TODO NOW: astype needed?
            img_bovw = img_bovw.astype(np.float32)
        else:
            start = time.time()
            result_tuple = self.get_descriptor(img_path, both=True, total_count_coordinate_for_bow=True)
            if len(result_tuple) != 3 or result_tuple[0] is None:
                message = f'Could not get descriptor for image {query_name}'
                raise ValueError(message)
            img_descriptor, img_bovw, kp = result_tuple
            img_bovw = img_bovw.astype(np.float32)

            time_getting_descriptor = round(time.time() - start, 3)
            logger.info("Descriptor for query got in {}".format(time_getting_descriptor))

        times += [('time_getting_descriptor', time_getting_descriptor)]

        start = time.time()

        assert(img_bovw.shape[0] == self.n_words + 1)

        start = time.time()
        bad_words = np.where(freqs > p_fine_max * data_dependent_params['count_images'])[0]
        time_computing_bad_words = round(time.time() - start, 3)
        times += [('time_computing_bad_words', time_computing_bad_words)]

        candidates = self.get_candidates(img_bovw[:-1], bad_words=bad_words)
        if len(candidates) == 0:
            logger.warning('0 candidates if filter. Trying increasing p_fine_max *2')
            p_fine_max = min(1.0, 2 * p_fine_max)
            # p_fine_max = 1.0
            bad_words = np.where(freqs > p_fine_max * data_dependent_params['count_images'])[0]
            candidates = self.get_candidates(img_bovw[:-1], bad_words=bad_words)

        time_retrieving_candidates = round(time.time() - start, 3)
        times += [('len(candidates)', len(candidates))]
        times += [('time_retrieving_candidates', time_retrieving_candidates)]
        times += [('new_query is not None', new_query is not None)]
        logger.info(f"{len(candidates)} candidates by words got in {time_retrieving_candidates}")

        if len(candidates) == 0:
            # Epilog
            candidates = [(c, 0) for c in candidates]
            result = candidates[:topk]
            time_getting_names = self.put_photo_names_to_result_return_time(result)
            times += [('time_getting_names', time_getting_names)]
            self.log_answers(query_name, result, sv_enable, qe_enable)
            self.log_search_times(times)
            return result


        # STEP 2. PRELIMINARY RANKING
        # TODO: Use heapq to obtain preliminary top n_candidates?

        def divide_sparse_on_vec(C, D):
            r, c = C.nonzero()
            rD_sp = sparse.csr_matrix(((1.0 / D)[r], (r, c)), shape=(C.shape))
            out = C.multiply(rD_sp)
            return out

        start = time.time()
        bow_candidates_without_last_col = self.bow[candidates, :-1]
        bow_candidates_last_col = self.bow[candidates, -1].toarray().squeeze()
        time_taking_bow_candidates = round(time.time() - start, 3)
        times += [('time_taking_bow_candidates', time_taking_bow_candidates)]

        start = time.time()
        bow = (divide_sparse_on_vec(bow_candidates_without_last_col, bow_candidates_last_col)
               .multiply(idf.reshape(1, -1)))
        ranks = euclidean_distances(
            bow,
            (img_bovw[:-1] / img_bovw[-1]).reshape(1, -1)
        )
        ranks = ranks.reshape((-1,))

        time_computing_ranks = round(time.time() - start, 3)
        times += [('time_computing_ranks', time_computing_ranks)]

        start = time.time()
        # TODO: try np.select_topk instead of sorting.
        ranks_sorted_args = np.argsort(ranks)[:n_candidates]
        candidates_chosen = np.array(candidates)[ranks_sorted_args]
        candidates = [(candidate_chosen_now, None) for candidate_chosen_now in candidates_chosen]
        time_preliminary_sorting = round(time.time() - start, 3)
        times += [('time_preliminary_sorting', time_preliminary_sorting)]
        logger.info("Short list got in {}".format(time_preliminary_sorting + time_computing_ranks + time_taking_bow_candidates + time_retrieving_candidates))

        # STEP 3. SPATIAL VERIFICATION
        if not sv_enable:
            # Epilog
            candidates = [(c, 0) for c in candidates]
            result = candidates[:topk]
            time_getting_names = self.put_photo_names_to_result_return_time(result)
            times += [('time_getting_names', time_getting_names)]
            self.log_answers(query_name, result, sv_enable, qe_enable)
            self.log_search_times(times)
            return result

        start = time.time()

        # NOTE: Refactor ransac in the future so that it does not return things for debug.
        # NOTE: candidates are rearranged (due to database_serivice specifics) and that is why ransac returns it.
        [candidates, all_matches, all_matches_masks, all_transforms,
         verified,
         descriptors_kp] = self.ransac(img_descriptor, kp, candidates,
                                       n_inliners_thr, max_verified)
        assert(len(candidates) == len(all_matches) == len(all_matches_masks) == len(all_transforms) == len(verified) == len(descriptors_kp))

        logger.info('Spatial verification got in {}s'.format(round(time.time() - start), 3))
        # if debug:
        #     n = 1
        #     draw_params = dict(matchColor=(0, 255, 0),
        #                        singlePointColor=(255, 0, 0),
        #                        matchesMask=all_matches_masks[n],
        #                        flags=0)
        #
        #     if type(img_path) is str:
        #         img1 = cv2.imread(img_path, 0)
        #     else:
        #         img1 = img_path
        #     img2 = cv2.imread(candidates[n][1])
        #
        #     h, w = img1.shape
        #     pts = np.float32([[0, 0],
        #                       [0, h - 1],
        #                       [w - 1, h - 1],
        #                       [w - 1, 0]]).reshape(-1, 1, 2)
        #     dst = cv2.perspectiveTransform(pts, all_transforms[n])
        #     img2 = cv2.polylines(img2, [np.int32(dst)],
        #                          True, (255, 50, 255), 30, cv2.LINE_AA)
        #
        #     img3 = cv2.drawMatches(img1, kp,
        #                            img2, descriptors_kp[n],
        #                            all_matches[n], None, **draw_params)
        #     logger.debug(img3.shape)
        #     img3[:, :, 0], img3[:, :, 2] = img3[:, :, 2], img3[:, :, 0]
        #     plt.imshow(img3)
        #     logger.debug(all_transforms)
        #     plt.show()

        sv_candidates = sorted([(candidates[i],
                                 np.count_nonzero(all_matches_masks[i]))
                                for i in range(len(verified))],
                               key=lambda x: x[1], reverse=True)

        # STEP 4. QUERY EXPANSION
        if qe_enable and new_query is None and np.count_nonzero(verified) < qe_limit:
            start = time.time()
            top_res = []
            for sv_candidate in sv_candidates[:qe_avg]:
                sv_candidate_bow = self.bow[sv_candidate[0][0]].toarray()
                top_res.append(sv_candidate_bow)

            one_more_query = (sum(top_res) + img_bovw) / (len(top_res) + 1)
            one_more_query = one_more_query.reshape((-1, ))
            new_sv_candidates = self.search(img_path, n_candidates,
                                            topk, n_inliners_thr, max_verified,
                                            qe_avg, qe_limit, one_more_query,
                                            qe_enable=qe_enable, sv_enable=sv_enable,
                                            debug=debug,
                                            precomputed_img_descriptor=img_descriptor,
                                            precomputed_kp=kp,
                                            query_name=query_name,
                                            p_fine_max=p_fine_max)

            old = set(sv_candidates[i][0][0] for i in range(len(sv_candidates)))
            new = set(new_sv_candidates[i][0][0] for i in range(len(new_sv_candidates)))
            duplicates = set(old) & set(new)
            new_sv_candidates = [el for el in new_sv_candidates
                                 if el[0][0] not in duplicates]
            sv_candidates = sorted(sv_candidates + new_sv_candidates,
                                   key=lambda x: x[1], reverse=True)
            logger.info("Query Expansion got in {}s".format(round(time.time() - start), 3))

        # Epilog
        result = sv_candidates[:topk]
        time_getting_names = self.put_photo_names_to_result_return_time(result)
        times += [('time_getting_names', time_getting_names)]
        self.log_answers(query_name, result, sv_enable, qe_enable)
        self.log_search_times(times)
        return result

    def ransac(self, img_descriptor, kp, candidates,
               min_inliners, max_verified):

        photo_descriptors_raw_iterator = database_service.get_photos_descriptors_by_rowids_iterator(
            self.db,
            [candidate[0]
             for candidate
             in candidates])
        names_descriptors_kp_pair_iterator = ((photo_descriptor_raw['rowid'],
                                               self.deserialize_descriptor(photo_descriptor_raw['descriptor']))
                                              for photo_descriptor_raw
                                              in photo_descriptors_raw_iterator)

        bf = cv2.BFMatcher(cv2.NORM_L2)

        MIN_MATCH_COUNT = 10  # 2 NOTE: smaller value for tests is better.
        all_matches = []
        all_matches_masks = []
        all_transforms = []
        candidates_rearranged = []
        verified = []
        n_verified = 0

        # Now it is for backward-compatibility. Remove in the future.
        descriptors_kp = []

        for i, rowid_descriptor_kp_pair in enumerate(names_descriptors_kp_pair_iterator):
            rowid = rowid_descriptor_kp_pair[0]
            descriptor = rowid_descriptor_kp_pair[1][0]
            descriptor_kp = rowid_descriptor_kp_pair[1][1]

            matches = bf.match(img_descriptor, descriptor)

            if len(matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([descriptor_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                descriptors_kp += [descriptor_kp]
                # One candidate is pair of (some_id, name) for backward-compatibility
                candidates_rearranged += [(rowid, None)]
                all_matches.append(matches)
                all_matches_masks.append(matchesMask)
                all_transforms.append(M)

                if np.count_nonzero(matchesMask) > min_inliners:
                    verified.append(1)
                    n_verified += 1
                else:
                    verified.append(0)
            else:
                logger.info("Not enough matches are found - {}/{}".format(len(matches),
                                                                           MIN_MATCH_COUNT))

            if n_verified == max_verified:
                break

        return [candidates_rearranged, all_matches, all_matches_masks, all_transforms,
                verified,
                descriptors_kp]

    # Very common
    @classmethod
    def get_storage_path(cls, database, name):
        return str(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database / name)

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
            sparse.save_npz(f, bow)

    @classmethod
    def _save_inverted_index(cls, database, name,
                             inverted_index):
        with open(cls.get_inv_path(database, name), 'wb') as f:
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
    def get_inv_path(cls, database, name):
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
        with open(CBIRCore.get_params_path(self.database, self.name), 'rb') as f:
            params = pickle.load(f)
        return params

    def load_data_dependent_params(self):
        with open(CBIRCore.get_data_dependent_params_path(self.database, self.name), 'rb') as f:
            data_dependent_params = pickle.load(f)
        return data_dependent_params

    def load_inv(self):
        with open(CBIRCore.get_inv_path(self.database, self.name), 'rb') as f:
            inverted_index = pickle.load(f)
        return inverted_index

    def load_index(self):
        with open(CBIRCore.get_des_path(self.database, self.name), 'rb') as f:
            index = pickle.load(f)
        index = {k: (v[0],
                     [cv2.KeyPoint(*el) for el in zip(v[1], v[2], v[3])])
                 for k, v in index.items()}
        return index

    def load_f_names(self):
        with open(CBIRCore.get_f_names_path(self.database, self.name), 'rb') as f:
            f_names = pickle.load(f)
        return f_names

    def load_bow(self):
        with open(CBIRCore.get_bow_path(self.database, self.name), 'rb') as f:
            bow = sparse.load_npz(f)
        return bow

    def load_freqs(self):
        with open(CBIRCore.get_data_dependent_params_path(self.database, self.name), 'rb') as f:
            freqs = pickle.load(f)['freqs']
        return freqs

    @classmethod
    def copy_descriptors_from_to(cls,
                                 from_database, from_name,
                                 to_database, to_name,
                                 to_index,
                                 for_training):
        """
        Copies descriptors to another index and setting new purpose values.
        """

        # database_service.copy_
        raise NotImplementedError

    @classmethod
    def descriptors_compatible(cls, database, first_name, second_name):
        from_des_type = CBIRCore(database, first_name).des_type
        to_des_type = CBIRCore(database, second_name).des_type
        return from_des_type != to_des_type

    @classmethod
    def get_databases(cls):
        return [
            filename
            for filename in os.listdir(cbir.DATABASES_RELATIVE_TO_BASE_DIR)
            if os.path.isdir(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / filename)]

    @classmethod
    def get_cbir_indexes_of_database(cls, database):
        return [
            filename
            for filename in os.listdir(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database)
            if os.path.isdir(Path(cbir.DATABASES_RELATIVE_TO_BASE_DIR) / database / filename)]

    def load_fd(self):
        max_keypoints = self.max_keypoints
        logger.info(f"Loading cbir_pretrained network {self.des_type}...")
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
        logger.debug('Setting fd')
        self.fd = fd

    def unset_fd(self):
        logger.debug('Unsetting fd')
        self.fd = None

    def load_ca(self):
        with open(CBIRCore.get_clusterer_path(self.database, self.name), 'rb') as f:
            ca = pickle.load(f)
        return ca

    def set_ca(self, ca):
        self.ca = ca

    def unset_ca(self):
        self.ca = None

    # Already defined
    # def load_bow(self):
    #     with open(CBIRCore.get_bow_path(self.database, self.name), 'rb') as f:
    #         bow = pickle.load(f)
    #     return bow

    def set_bow(self, bow):
        self.bow = bow

    def unset_bow(self):
        self.bow = None

    # Already defined
    # def load_inv(self):
    #     with open(CBIRCore.get_inv_path(self.database, self.name), 'rb') as f:
    #         inv = pickle.load(f)
    #     return inv

    def set_inv(self, inv):
        self.inv = inv

    def unset_inv(self):
        self.inv = None

    def load_most_frequent(self):
        return self.load_data_dependent_params()['most_frequent']

    def set_most_frequent(self, most_frequent):
        self.most_frequent = most_frequent

    def unset_most_frequent(self):
        self.most_frequent = None


    def serialize_descriptor(self, descriptor):
        # return pickle.dumps((descriptor[0],
        #                      [p.pt[0] for p in descriptor[1]],
        #                      [p.pt[1] for p in descriptor[1]],
        #                      [p.size for p in descriptor[1]]),
        #                     protocol=pickle.HIGHEST_PROTOCOL)
        return pickle.dumps((descriptor[0],
                             [(p.pt[0], p.pt[1], p.size) for p in descriptor[1]]),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_descriptor(self, descriptor):
        descriptor = pickle.loads(descriptor)
        return (descriptor[0], [cv2.KeyPoint(*el) for el in descriptor[1]])

    decorator_load_ca_if_needed = staticmethod(decorator_load_ca_if_needed)

    decorator_load_fd_if_needed = staticmethod(decorator_load_fd_if_needed)

    decorator_load_bow_if_needed = staticmethod(decorator_load_bow_if_needed)

    decorator_load_inv_if_needed = staticmethod(decorator_load_inv_if_needed)

    decorator_load_most_frequent_if_needed = staticmethod(decorator_load_most_frequent_if_needed)


def compute_idf_lazy(freqs, total_count_documents):
    # not_zero = np.where(freqs != 0)[0]
    # idf = np.zeros(freqs.shape[0])
    # idf[not_zero] = np.log(total_count_documents / freqs[not_zero])
    idf = np.log(total_count_documents / (1 + freqs))
    return idf
