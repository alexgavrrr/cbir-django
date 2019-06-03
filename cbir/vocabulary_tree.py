import logging

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pqkmeans


class VocabularyTree:
    def __init__(self, L=4, K=10):
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

    def fit(self, data):
        # TODO: `data_loader` param instead of `data`

        # batch_size = max(1000, self.K * 100)

        self.encoder = pqkmeans.encoder.PQEncoder(num_subdim=4, Ks=256)
        self.encoder.fit(data)
        pq_data = self.encoder.transform(data)

        self.ca = []
        model = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=self.K)
        model.fit(pq_data)
        self.ca.append(model)

        labels = self.ca[0].predict(pq_data).astype(np.int32)
        for l in range(1, self.L):
            print("At level {}".format(l))
            ca_l = []
            n_c = self.K ** l
            for i in range(n_c):
                idx = np.where(labels == i)[0]

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
