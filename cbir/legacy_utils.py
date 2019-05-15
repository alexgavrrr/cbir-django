import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN

# resize image to size 32x32
cv2_scale36 = lambda x: cv2.resize(x, dsize=(36, 36),
                                   interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def find_image_files(root, extensions, recursive=True):
    files = []
    for file_dir in os.listdir(root):
        full_path = os.path.join(root, file_dir)
        if os.path.isdir(full_path) and recursive:
            files += find_image_files(full_path, extensions, recursive)

        # TODO: Consider whether this code can add dir to list of files.
        for ext in extensions:
            if file_dir.endswith(ext):
                files.append(full_path)
                break
    return sorted(files)  # sort files in ascend order to keep relations


def find_image_files_bounded(root, extensions, recursive=True, max_images=10000):
    files = []
    for file_dir in os.listdir(root):
        full_path = os.path.join(root, file_dir)
        if os.path.isdir(full_path) and recursive and len(files) < max_images:
            files += find_image_files(full_path, extensions, recursive)

        # TODO: Consider whether this code can add dir to list of files.
        for ext in extensions:
            if file_dir.endswith(ext):
                files.append(full_path)
                break

        if len(files) >= max_images:
            return sorted(files)[:max_images]

    return sorted(files)[:max_images]  # sort files in ascend order to keep relations


def draw_patches(patches, nc=5):
    nr = len(patches) // nc
    if len(patches) % nc:
        nr += 1

    plt.figure(figsize=(8, 6))
    ax = [plt.subplot(nr, nc, i + 1) for i in range(len(patches))]
    for i, a in enumerate(ax):
        a.axis('off')
        a.imshow(patches[i])
        # a.set_aspect('equal')

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()


def draw_result(query, similar, nc=5):
    if query is not None:
        img = cv2.imread(query)

    nr = len(similar) // nc
    if len(similar) % nc:
        nr += 1

    sim_images = [cv2.imread(f) for f in similar]

    if query is not None:
        plt.figure(1)
        plt.title('Query')
        plt.imshow(img)

    plt.figure(figsize=(8, 6))
    ax = [plt.subplot(nr, nc, i + 1) for i in range(len(sim_images))]

    for i, a in enumerate(ax):
        a.axis('off')
        a.imshow(sim_images[i])
        # a.set_aspect('equal')

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()


def estimate_with_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))


def dist(kp1, kp2):
    x1, y1 = kp1.pt[0], kp1.pt[1]
    x2, y2 = kp2.pt[0], kp2.pt[1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def filter_points(kp, min_distance=17):
    n_kp = len(kp)
    deleted = [False for i in range(n_kp)]
    for i in range(n_kp):
        if deleted[i]:
            continue
        for j in range(i + 1, n_kp):
            if not deleted[j] and dist(kp[i], kp[j]) < min_distance:
                deleted[j] = True
    return [kp[i] for i in range(n_kp) if not deleted[i]]


def filter_points_ann(kp, min_distance=17):
    n_kp = len(kp)
    deleted = [False for i in range(n_kp)]
    flann = pyflann.FLANN()
    flann.set_distance_type('euclidean')
    params = flann.build_index(kp, algorithm="kmeans",
                               target_precision=0.9, log_level="info")
    print(params)
    result, dists = flann.nn_index(kp, 50, checks=params["checks"])
    print(result)
    print(dists)

    return [kp[i] for i in range(n_kp) if not deleted[i]]
