

import argparse
import os
import numpy as np
import torch
from models.AutoEncoder import AE_Aug
from models.DCGAN import DCGAN_Aug
from utils import all_data


class DataAugment():
    def __init__(self, X, y, model_idx) -> None:
        self.X = X
        self.y = y
        self.model_idx = model_idx
    
    def runAugmentation(self,args,mode):
        if args.aug_type == 0:
            # no augmentation
            return self.X, self.y
        elif args.aug_type == 1:
            # data sampling
            data_size = int(self.X.shape[0] * args.aug_ratio)
            aug_data = []
            for i in range(self.X.shape[1]):
                mu = torch.mean(self.X[:,i])
                std = torch.std(self.X[:,i])
                samples = torch.randn(data_size) * std + mu
                aug_data.append(samples)
            aug_data = torch.stack(aug_data, dim=1)
            return aug_data
        elif args.aug_type == 2:
            # noisy
            data_size = int(self.X.shape[0] * args.aug_ratio)
            aug_data = []
            for i in range(self.X.shape[1]):
                mu = torch.mean(self.X[:,i])
                std = torch.std(self.X[:,i])
                samples = torch.randn(data_size) * std + mu
                aug_data.append(samples)
            noisy_data = torch.stack(aug_data, dim=1)
            ran_idx = np.random.choice(np.arange(self.X.shape[0]),data_size)
            original_data = self.X[ran_idx]
            aug_data = noisy_data * 0.2 + original_data * 0.8
            return aug_data
        elif args.aug_type == 3:
            # DCGAN
            aug_data = DCGAN_Aug(mode='train1',data=self.X)
            print(self.X[:5,:10])
            print(aug_data[:5,:10])
            exit()
        elif args.aug_type == 4:
            # AutoEncoder
            num_data = int(args.aug_ratio * self.X.shape[0])
            aug_data = AE_Aug(mode=mode,data=self.X,num_data=num_data,model_idx=self.model_idx)
            return aug_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_type", type=int, default=4)
    parser.add_argument("--aug_ratio", type=float, default=1.0)
    parser.add_argument('-m','--mode',type=str,default='train')
    args = parser.parse_args()

    X_all,y_all = all_data()
    for id in range(1,16):
        X_id = X_all[id-1,:,:].reshape(-1, 310)
        y_id = y_all[id-1].view(-1)
        num = int(X_id.shape[0]*1)    
        X_tr = X_id[:num, :]
        y_tr = y_id[:num]
        aug_data_all = []
        aug_label_all = []
        os.makedirs('aug_data/',exist_ok=True)
        os.makedirs(f'aug_data/aug_type{args.aug_type}',exist_ok=True)
        
        for i in range(4):
            label_idx = np.where(np.array(y_tr)==i)[0]
            x_label = X_tr[label_idx]
            y_label = y_tr[label_idx]
            aug = DataAugment(x_label,y_label,model_idx=f'{id}_{i}')
            if  args.mode == 'train':
                aug.runAugmentation(args=args,mode=args.mode)
                exit(0)
            else:
                aug_data = aug.runAugmentation(args=args,mode=args.mode)
                aug_data_all.append(aug_data)
                aug_label_all.append(y_label)
        aug_data_all = torch.cat(aug_data_all)
        aug_label_all = torch.cat(aug_label_all)
        print(id,aug_data_all.shape,aug_label_all.shape)
        np.save(f'aug_data/aug_type{args.aug_type}/X{id}.npy',np.array(aug_data_all))
        np.save(f'aug_data/aug_type{args.aug_type}/y{id}.npy',np.array(aug_label_all))
            


    