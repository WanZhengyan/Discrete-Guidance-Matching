import torch
# import gym
# import d4rl
import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

# Dataset iterator
def inf_train_gen(data, batch_size=200):
    # print(data)
    if data == "swissroll":
        # print(data)
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        energy = np.sum(data**2, axis=-1, keepdims=True) / 9.0 - 1.5
        # Clip and scale the data
        data = np.clip(data, -4, 4)
        data = data * 4
        data = data + 16
        data = np.round(data).astype(int)
        return data, energy
    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)


        center_dist = X[:,0]**2 + X[:,1]**2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >=8.5)] = 0.667 
        energy[(center_dist >=5.0) & (center_dist <8.5)] = 0.333 
        energy[(center_dist >=2.0) & (center_dist <5.0)] = 1.0 
        energy[(center_dist <2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        # Clip and scale the data
        X = np.clip(X, -4, 4)
        X = X * 4
        X = X + 16
        X = np.round(X).astype(int)
        return X.astype("float32"), energy[:,None] - 1.0

    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])

        # Clip and scale the data
        data = np.clip(data, -4, 4)
        data = data * 4
        data = data + 16
        data = np.round(data).astype(int)
        return data.astype(np.float32), (y > 0.5).astype(np.float32)[:,None] - 1.0

    elif data == "8gaussians":
        scale = 4.
        centers = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                    (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        indexes = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            indexes.append(idx)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414

        # Clip and scale the data
        dataset = np.clip(dataset, -4, 4)
        dataset = dataset * 4
        dataset = dataset + 16
        dataset = np.round(dataset).astype(int)
        return dataset, np.array(indexes, dtype="float32")[:,None] / 7.0 - 1.0

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1

        # Clip and scale the data
        x = np.clip(x, -4, 4)
        x = x * 4
        x = x + 16
        x = np.round(x).astype(int)
        return x, np.clip((1-np.concatenate([n,n]) / 10),0,1) - 1.0

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        points_x = points[:,0]
        judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))

        # Clip and scale the data
        points = np.clip(points, -4, 4)
        points = points * 4
        points = points + 16
        points = np.round(points).astype(int)
        return points, judger.astype(np.float32)[:,None] - 1.0 
    else:
        assert False

class Toy_dataset(torch.utils.data.Dataset):
    def __init__(self, name, datanum=1000000, device="cuda" if torch.cuda.is_available() else "cpu"):
        assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
        self.datanum =datanum
        self.name = name
        self.datas, self.energy = inf_train_gen(name, batch_size=datanum)
        self.datas = torch.Tensor(self.datas).long().to(device)
        self.energy = torch.Tensor(self.energy).to(device)
        self.datadim = 2
      
    def __getitem__(self, index):
        return {"a": self.datas[index], "e": self.energy[index]} # a: data, e: energy

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.datanum


