import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import sys
sys.path.append("..")
from models.ADDA import MLP, Classifier
from utils import all_data
from sklearn.metrics import accuracy_score
import os
from sklearn import svm


def main(args, id, save_path):
    # load data
    X_all,y_all = all_data()
    indice = list(range(15))
    indice.remove(id-1)
    X_train = X_all[indice,:,:].reshape(-1, 310)
    y_train = y_all[indice].view(-1)
    X_test = X_all[id-1].reshape(-1, 310)
    y_test = y_all[id-1].view(-1)

    svm_model = svm.SVC(kernel='linear', C=1.0, gamma='auto',max_iter=2000)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test on person {i}, acc: {acc:.4f}",flush=True)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder_units", type=int, nargs='+', default=[310, 256, 128])
    parser.add_argument("--predictor_units", type=int, nargs='*', default=[310, 64, 32])
    parser.add_argument("--discriminator_units", type=int, nargs='*', default=[128, 48])
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('-r',"--ratio", type=float, default=1.0)
    parser.add_argument('-t',"--aug_type", type=int, default=4)
    parser.add_argument('-s',"--data_source", type=int, default=0)
    args = parser.parse_args()

    acc = []
    for i in range(1, 16):
        cur_acc = main(args, i, save_path=f"../ckpts/mlp/{i}/")
        acc.append(cur_acc)
    print(f"average acc: {np.mean(acc)}",flush=True)
