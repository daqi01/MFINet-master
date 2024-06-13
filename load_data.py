import torch
import numpy as np
import os
import h5py
import glob
from torch.utils.data import Dataset

#from sklearn.neighbors import NearestNeighbors
#from scipy.spatial.distance import minkowski
#from scipy.spatial import cKDTree

def load_data(train, use_normals):
    if train:
        partition = 'train'
    else:
        partition = 'test'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
        else: data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

# def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
#     pointcloud1 = pointcloud1
#     num_points = pointcloud1.shape[0]
#     nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
#                              metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
#     random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
#     idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
#     gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
#     return pointcloud1[idx1, :], gt_mask

# def random_select_points(pc, m = 768):
#     if m < 0:
#         idx = np.arange(pc.shape[0])
#         np.random.shuffle(idx)
#         return pc[idx, :]
#     n = pc.shape[0]
#     replace = False if n >= m else True
#     idx = np.random.choice(n, size=(m, ), replace=replace)
#     return pc[idx, :]
def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)
def random_crop(pc, p_keep=0.75):
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(pc[:, :3], axis=0)
    pc_centered = pc[:, :3] - centroid

    dist_from_plane = np.dot(pc_centered, rand_xyz)
    mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
    return pc[mask, :]

def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
    # N, C = pointcloud.shape
    sigma = sigma*np.random.random_sample()
    pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
    return pointcloud

class ModelNet40Data(Dataset):
    def __init__(
        self,
        train=False,
        num_points=1024,
        randomize_data=False,
        use_normals=False,
    ):
        super(ModelNet40Data, self).__init__()
        self.data, self.labels = load_data(train, use_normals)
        if not train: self.shapes = self.read_classes_ModelNet40()
        self.num_points = num_points
        self.randomize_data = randomize_data

    def __getitem__(self, idx):
        if self.randomize_data:
            src_points = self.randomize(idx)
            tgt_points = self.randomize(idx)
        else:
            src_points = self.data[idx].copy()
            tgt_points = self.data[idx].copy()

        src_points = torch.from_numpy(src_points[:self.num_points, :]).float()
        #两次采样
        tgt_points = torch.from_numpy(tgt_points[:self.num_points, :]).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        return src_points, tgt_points, label

    def __len__(self):
        return self.data.shape[0]

    def randomize(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx, pt_idxs].copy()

    def get_shape(self, label):
        return self.shapes[label]

    def read_classes_ModelNet40(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
        shape_names = file.read()
        shape_names = np.array(shape_names.split('\n')[:-1])
        return shape_names

class RegistrationData(Dataset):
    def __init__(self, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise_source=False, noise_template=False):
        super(RegistrationData, self).__init__()

        self.set_class(data_class)
        self.partial_template = partial_template
        self.partial_source = partial_source
        self.noise_source = noise_source
        self.noise_template = noise_template
        from ops.transform_functions import PCRNetTransform
        self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=0.5)

    def __len__(self):
        return len(self.data_class)

    def set_class(self, data_class):
        self.data_class = data_class

    def __getitem__(self, index):
        source, template, label = self.data_class[index]
        self.transforms.index = index  # for fixed transformations in PCRNet.
        source = self.transforms(source)
        igt = self.transforms.igt

        if self.partial_source:
            #source, self.source_mask = farthest_subsample_points(source)
            source = random_crop(source.numpy())
            source = torch.from_numpy(source).float()
        if self.partial_template:
            #template, self.template_mask = farthest_subsample_points(template)
            template = random_crop(template.numpy())
            template = torch.from_numpy(template).float()

        if self.noise_source:
            source = jitter_pointcloud(source,sigma=0.04)
        if self.noise_template:
            template = jitter_pointcloud(template,sigma=0.04)

        return template, source, igt

if __name__ == '__main__':
    class Data():
        def __init__(self):
            super(Data, self).__init__()
            self.data, self.label = self.read_data()

        def read_data(self):
            return [4,5,6], [4,5,6]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.label[idx]

    cd = RegistrationData()

    #import ipdb; ipdb.set_trace()
