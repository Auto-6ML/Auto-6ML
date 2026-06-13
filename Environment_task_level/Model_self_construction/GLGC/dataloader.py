from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class DHA(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path)['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path)['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_views(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)



class Land(Dataset):
    def __init__(self, path):
        print('LandUse-21')
        data2 = scipy.io.loadmat(path)['X'][0, 1].astype(np.float32)
        data3 = scipy.io.loadmat(path)['X'][0, 2].astype(np.float32)
        labels = scipy.io.loadmat(path)['Y']
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x2.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(
            self.x2[idx]), torch.from_numpy(
            self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_views(self, new_data):
        """
        Update the value of self. view
        """
        self.x2 = new_data[0].numpy().astype(np.float32)
        self.x3 = new_data[1].numpy().astype(np.float32)


class ProteinFold(Dataset):
    def __init__(self, path):
        print('ProteinFold')
        data1 = scipy.io.loadmat(path)['X'][0, 0].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X'][1, 0].astype(np.float32)
        data3 = scipy.io.loadmat(path)['X'][2, 0].astype(np.float32)
        data4 = scipy.io.loadmat(path)['X'][3,0].astype(np.float32)
        data5 = scipy.io.loadmat(path)['X'][4,0].astype(np.float32)
        data6 = scipy.io.loadmat(path)['X'][5,0].astype(np.float32)
        data7 = scipy.io.loadmat(path)['X'][6,0].astype(np.float32)
        data8 = scipy.io.loadmat(path)['X'][7,0].astype(np.float32)
        data9 = scipy.io.loadmat(path)['X'][8,0].astype(np.float32)
        data10 = scipy.io.loadmat(path)['X'][9,0].astype(np.float32)
        data11 = scipy.io.loadmat(path)['X'][10,0].astype(np.float32)
        data12 = scipy.io.loadmat(path)['X'][11,0].astype(np.float32)
        labels = scipy.io.loadmat(path)['y']
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.x7 = data7
        self.x8 = data8
        self.x9 = data9
        self.x10 = data10
        self.x11 = data11
        self.x12 = data12
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.x1[idx]),
            torch.from_numpy(self.x2[idx]),
            torch.from_numpy(self.x3[idx]),
            torch.from_numpy(self.x4[idx]),
            torch.from_numpy(self.x5[idx]),
            torch.from_numpy(self.x6[idx]),
            torch.from_numpy(self.x7[idx]),
            torch.from_numpy(self.x8[idx]),
            torch.from_numpy(self.x9[idx]),
            torch.from_numpy(self.x10[idx]),
            torch.from_numpy(self.x11[idx]),
            torch.from_numpy(self.x12[idx]),
        ], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_views(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        self.x3 = new_data[2].numpy().astype(np.float32)
        self.x4 = new_data[3].numpy().astype(np.float32)
        self.x5 = new_data[4].numpy().astype(np.float32)
        self.x6 = new_data[5].numpy().astype(np.float32)
        self.x7 = new_data[6].numpy().astype(np.float32)
        self.x8 = new_data[7].numpy().astype(np.float32)
        self.x9 = new_data[8].numpy().astype(np.float32)
        self.x10 = new_data[9].numpy().astype(np.float32)
        self.x11 = new_data[10].numpy().astype(np.float32)
        self.x12 = new_data[11].numpy().astype(np.float32)


class ALOI(Dataset):
    def __init__(self, path):
        print('ALOI')
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(scipy.io.loadmat(path)['X'][0, 0].astype(np.float32))
        data2 = scaler.fit_transform(scipy.io.loadmat(path)['X'][1, 0].astype(np.float32))
        data3 = scaler.fit_transform(scipy.io.loadmat(path)['X'][2, 0].astype(np.float32))
        data4 = scaler.fit_transform(scipy.io.loadmat(path)['X'][3, 0].astype(np.float32))
        labels = scipy.io.loadmat(path)['y']
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(
            self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(
            self.x3[idx]), torch.from_numpy(
            self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_views(self, new_data):
        """
        Update the value of self. view
        """
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        self.x3 = new_data[2].numpy().astype(np.float32)
        self.x4 = new_data[3].numpy().astype(np.float32)


def load_data(dataset):
    if dataset == "DHA":
        dataset = DHA('data/DHA.mat')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == "Land":
        dataset = Land('data/LandUse-21.mat')
        dims = [59, 40]
        view = 2
        data_size = 2100
        class_num = 21
    elif dataset == "ProteinFold":
        dataset = ProteinFold('data/ProteinFold.mat')
        dims = [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27]
        view = 12
        data_size = 694
        class_num = 27
    elif dataset == "ALOI":
        dataset = ALOI('data/ALOI.mat')
        dims = [77, 13, 64, 125]
        view = 4
        data_size = 10800
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
