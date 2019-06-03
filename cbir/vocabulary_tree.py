import logging
import time

import numpy as np
import pqkmeans
from tqdm import tqdm


class VocabularyTree:
    def __init__(self, L=4, K=50):
        '''
        L - number of levels
        K - number of clusters at each level
        '''
        self.L = L
        self.K = K
        self.max_clusters = K ** L
        self.ca = None
        self.n_images = None
        self.encoder = None

    def fit(self,
            data_loader,
            sift1b_encoder,
            sift1b_pqcodes_path,
            dtype
            ):
        M = 4
        logger = logging.getLogger()

        start_first_stage = time.time()

        pq_data_first_part = None

        if sift1b_encoder:
            logger.info('Using pretrained encoder')
            self.encoder = sift1b_encoder
        else:
            logger.info('Building encoder')
            self.encoder = pqkmeans.encoder.PQEncoder(num_subdim=4, Ks=256)
            MAX_COUNT_SAMPLES_FOR_TRAINING_ENCODER = 10 ** 6
            SIZE_DESCRIPTOR = 128
            start = time.time()
            data_preloaded = np.empty(
                [MAX_COUNT_SAMPLES_FOR_TRAINING_ENCODER, SIZE_DESCRIPTOR], dtype=dtype)
            time_allocating_empty_array = round(time.time() - start, 3)
            logger.info(f'time_allocating_empty_array [{MAX_COUNT_SAMPLES_FOR_TRAINING_ENCODER}, {SIZE_DESCRIPTOR}]: {time_allocating_empty_array}')

            start = time.time()
            n_preloaded = 0
            for vec in data_loader:
                data_preloaded[n_preloaded, :] = vec
                n_preloaded += 1
                if n_preloaded == MAX_COUNT_SAMPLES_FOR_TRAINING_ENCODER:
                    break
            time_preloading_for_fitting_pq_encoder = round(time.time() - start, 3)
            logger.info(f'time_preloading_for_fitting_pq_encoder: {time_preloading_for_fitting_pq_encoder}, n_preloaded = {n_preloaded}')

            data_preloaded.resize([n_preloaded, SIZE_DESCRIPTOR])

            start = time.time()
            self.encoder.fit(data_preloaded)
            time_fitting_encoder = round(time.time() - start, 3)
            logger.info(f'time_fitting_encoder: {time_fitting_encoder}')

            start = time.time()
            pq_data_first_part = self.encoder.transform(data_preloaded)
            time_computing_pqcodes_1 = round(time.time() - start, 3)
            logger.info(f'time_computing_pqcodes_1: {time_computing_pqcodes_1}')
            del data_preloaded

        start = time.time()
        pq_data_second_part = []
        n_second_part = 0
        # for vec in tqdm(data_loader, desc='Computing pqcodes_2 iteratively'):
        #     # vectorize via batches?
        #     n_second_part += 1
        #     pq_data_second_part += [self.encoder.transform(vec.reshape((1, -1)))]
        pqcode_generator = self.encoder.transform_generator(data_loader)
        for code in tqdm(pqcode_generator):
            n_second_part += 1
            pq_data_second_part += [code]

        pq_data_second_part = np.array(pq_data_second_part) if len(pq_data_second_part) > 0 else np.empty(shape=[0, M])
        pq_data_second_part = pq_data_second_part.astype('uint8')
        time_computing_pqcodes_2 = round(time.time() - start, 3)
        logger.info(f'time_computing_pqcodes_2: {time_computing_pqcodes_2}')


        if pq_data_first_part is not None:
            pq_data_no_sift1b = np.vstack([pq_data_first_part, pq_data_second_part])
            del pq_data_first_part, pq_data_second_part
        else:
            pq_data_no_sift1b = pq_data_second_part
            del pq_data_second_part
        print(f'BBB pq_data.shape: {pq_data_no_sift1b.shape}')
        print(f'BBB pq_data.dtype: {pq_data_no_sift1b.dtype}')

        if sift1b_pqcodes_path is not None:
            start = time.time()
            if isinstance(sift1b_pqcodes_path, str):
                pq_data_sift1b = np.load(sift1b_pqcodes_path)
            else:
                pq_data_sift1b = sift1b_pqcodes_path
            pq_data = np.vstack([pq_data_no_sift1b, pq_data_sift1b])
            time_loading_sift1b_and_stacking_pq_datas = round(time.time() - start)
            logger.info(f'time_loading_sift1b_and_stacking_pq_datas: {time_loading_sift1b_and_stacking_pq_datas}')
            del pq_data_no_sift1b, pq_data_sift1b
        else:
            pq_data = pq_data_no_sift1b
            del pq_data_no_sift1b

        time_first_stage = round(time.time() - start_first_stage, 3)
        logger.info(f'time_first_stage: {time_first_stage}')

        to_resize = min((self.K ** self.L) * 1000, pq_data.shape[0])
        pq_data.resize((to_resize, pq_data.shape[1]))

        logger.info('Second stage: building tree clusterer...')

        sample_size_to_train_max = self.K * 1000
        idx = np.arange(pq_data.shape[0])
        if idx.shape[0] > sample_size_to_train_max:
            idx = np.random.choice(
                idx,
                size=sample_size_to_train_max,
                replace=True)

        self.ca = []
        model = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=self.K)
        start = time.time()
        logger.info('Fitting root clusterer...')
        model.fit(pq_data[idx, :])
        time_fitting_root_clusterer = round(time.time() - start, 3)
        logger.info(f'time_fitting_root_clusterer: {time_fitting_root_clusterer}')
        self.ca.append(model)

        start = time.time()
        logger.info(f'Predicting labels after root clusterer for all data {pq_data.shape}')
        labels = self.ca[0].predict(pq_data).astype(np.int32)
        time_predicting_root_labels_for_all_data = round(time.time() - start, 3)
        logger.info(f'time_predicting_root_labels_for_all_data: {time_predicting_root_labels_for_all_data}')


        for l in range(1, self.L):
            print("At level {}".format(l))
            ca_l = []
            n_c = self.K ** l
            for i in tqdm(range(n_c), desc=f'On level {l}'):
                idx = np.where(labels == i)[0]
                if idx.shape[0] > sample_size_to_train_max:
                    idx = np.random.choice(
                        idx,
                        size=sample_size_to_train_max,
                        replace=True)

                if len(idx) > self.K * 3:
                    model = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=self.K)
                    model.fit(pq_data[idx, :])
                    ca_l.append(model)
                    labels[idx] = labels[idx] * self.K + ca_l[i].predict(pq_data[idx, :]).astype(np.int32)
                else:
                    ca_l.append(None)
                    labels[idx] = labels[idx] * self.K

            self.ca.append(ca_l)

        return self

    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)

        pq_data = self.encoder.transform(data)
        c = self.ca[0].predict(pq_data).astype(np.int32)

        for l in range(1, self.L):
            c_set = set(c)
            for i in c_set:
                idx = np.where(c == i)[0]

                if self.ca[l][i] is None:
                    c[idx] *= self.K
                else:
                    c[idx] = c[idx] * self.K + self.ca[l][i].predict(pq_data[idx, :])

        return c
