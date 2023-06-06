import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def normalizer(vec, mean=None, std=None):
    if mean is None or std is None:
        return (vec-torch.mean(vec, dim=0))/torch.std(vec, dim=0)
    return (vec-mean)/std


def read_file(session, index, train=True):
    fpath = os.path.join("../SEED-IV", str(session))
    fpath = os.path.join(fpath, os.listdir(fpath)[index-1])
    ftype = "train_" if train else "test_"
    fdata = os.path.join(fpath, ftype+"data.npy")
    flabel = os.path.join(fpath, ftype+"label.npy")
    data = np.load(fdata).reshape(-1, 62*5)
    return torch.tensor(data).to(torch.float32), torch.tensor(np.load(flabel)).to(torch.long)


def all_data():
    X = []
    y = []
    for i in range(1, 16):
        person_X = []
        person_y = []
        for s in range(1,4):
            X_tr, y_tr = read_file(s, i, train=True)
            X_te, y_te = read_file(s, i, train=False)
            person_X.append(torch.cat((X_tr, X_te)))
            person_y.append(torch.cat((y_tr, y_te)))
        X.append(normalizer(torch.cat(person_X)))
        y.append(torch.cat(person_y))

    return torch.stack(X), torch.stack(y)

def aug_data(aug_type):
    path = f'aug_data/aug_type{aug_type}/'
    X_all = []
    y_all = []
    for id in range(1,16):
        x_path = path + f'X{id}.npy'
        y_path = path + f'y{id}.npy'
        X_all.append(torch.tensor(np.load(x_path)))
        y_all.append(torch.tensor(np.load(y_path)))

    return torch.stack(X_all), torch.stack(y_all)


class TrainDataset(Dataset):
    def __init__(self, id, ratio=1, data_source=0, aug_type=0):
        if data_source == 0:
            X_all, y_all = all_data()
        elif data_source == 1:
            X_all, y_all = aug_data(aug_type)
        else:
            x1,y1 = all_data()
            x2,y2 = aug_data(aug_type)
            X_all = torch.cat((x1,x2),dim=1)
            y_all = torch.cat((y1,y2),dim=1)

        indice = list(range(15))
        indice.remove(id-1)
        X_all = X_all[indice,:,:].reshape(-1, 310)
        y_all = y_all[indice].view(-1)
        num = int(X_all.shape[0]*1)
        if ratio == 1:
            self.X_tr = X_all[:num, :]
            self.y_tr = y_all[:num]
        else:
            assert ratio>0 and ratio<1
            idx = np.arange(num)
            np.random.shuffle(idx)
            idx = idx[:int(num*ratio)]
            self.X_tr = X_all[idx, :]
            self.y_tr = y_all[idx]
        

    def __getitem__(self, index):
        return self.X_tr[index], self.y_tr[index]
    
    def __len__(self):
        return self.X_tr.shape[0]

class TestDataset(Dataset):
    def __init__(self, id):
        X_all, y_all = all_data()

        self.X_te = X_all[id-1,:,:]
        self.y_te = y_all[id-1,:]

    def __getitem__(self, index):
        return self.X_te[index], self.y_te[index]
    
    def __len__(self):
        return self.X_te.shape[0]


def load_data(i, args):
    train_set = TrainDataset(i,args.ratio,args.data_source,args.aug_type)
    test_set = TestDataset(i)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)    
    return train_loader, test_loader


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight.data)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambd
        return grad_input, None
    

class DG_TrainDataset(Dataset):
    def __init__(self, id, ratio=1):
        X_all, y_all = all_data()
        indice = list(range(15))
        indice.remove(id-1)

        X_all = X_all[indice,:,:].reshape(-1, 310)
        y_all = y_all[indice].view(-1)
        num = int(X_all.shape[0]*1)

        self.ratio = ratio
        if ratio == 1:
            self.X_tr = X_all[:num, :]
            self.y_tr = y_all[:num]
        else:
            assert ratio>0 and ratio<1
            index = []
            for i in range(14):
                idx = np.arange(2505) + i*2505
                np.random.shuffle(idx)
                index.extend(list(idx[:int(2505*ratio)]))
            self.X_tr = X_all[index, :]
            self.y_tr = y_all[index]

    def __getitem__(self, index):
        return self.X_tr[index], self.y_tr[index], index//int(2505*self.ratio)
    
    def __len__(self):
        return self.X_tr.shape[0]



def load_dg_data(i, args):
    train_set = DG_TrainDataset(i,args.ratio)
    print('len of train set:',len(train_set))
    test_set = TestDataset(i)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)    
    return train_loader, test_loader

class ResTestDataset(Dataset):
    def __init__(self, id, ratio=1):
        X_all, y_all = all_data()

        self.X_te = X_all[id-1,:,:]
        self.y_te = y_all[id-1,:]

        if ratio != 1:
            assert ratio>0 and ratio<1
            num = self.X_te.shape[0]
            idx = np.arange(num)
            np.random.shuffle(idx)
            idx = idx[:int(num*ratio)]
            self.X_te = self.X_te[idx]
            self.y_te = self.y_te[idx]

    def __getitem__(self, index):
        return self.X_te[index], self.y_te[index]
    
    def __len__(self):
        return self.X_te.shape[0]

def load_res_data(i, args, train_flag=True):
    if train_flag:
        data_set = ResTestDataset(i, args.ratio)
    else:
        data_set = TestDataset(i)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    return data_loader
    
