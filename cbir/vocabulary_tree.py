import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


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

        # TODO: Ensure that this is redundant and delete.
        self.inverted_index = [[[], []] for i in range(self.max_clusters)]

        self.n_images = None

    def fit(self, data):
        self.ca = []
        self.ca.append(MiniBatchKMeans(n_clusters=self.K,
                                       batch_size=1000).fit(data))
        labels = self.ca[0].predict(data).astype(np.int32)

        for l in range(1, self.L):
            print("At level {}".format(l))
            ca_l = []
            n_c = self.K ** l
            for i in range(n_c):
                idx = np.where(labels == i)[0]

                if len(idx) > self.K * 3:
                    ca_l.append(MiniBatchKMeans(n_clusters=self.K,
                                                batch_size=1000,
                                                init_size=self.K * 3).fit(data[idx, :]))
                    labels[idx] = labels[idx] * self.K + ca_l[i].predict(data[idx, :]).astype(np.int32)
                else:
                    ca_l.append(None)
                    labels[idx] = labels[idx] * self.K

            self.ca.append(ca_l)

        return self

    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)

        c = self.ca[0].predict(data).astype(np.int32)

        for l in range(1, self.L):
            c_set = set(c)
            for i in c_set:
                idx = np.where(c == i)[0]

                if self.ca[l][i] is None:
                    c[idx] *= self.K
                else:
                    c[idx] = c[idx] * self.K + self.ca[l][i].predict(data[idx, :])

        return c
