import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm, trange
import sys
sys.path.append("..")
from models.ADDA import MLP, Classifier
from utils import load_data, seed_all, weight_init, GRL
from sklearn.metrics import accuracy_score
import os


def evaluate(model, data_loader, device):
    model.eval()
    acc = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.detach().cpu().numpy().astype('float64')
            pred = np.argmax(pred, axis=-1)
            y = y.detach().cpu().numpy()

            accuracy = accuracy_score(y, pred)
            acc.append(accuracy)
    acc = np.mean(acc)
    return acc


def main(args, i, save_path):
    seed_all(42)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    # load data
    source_loader, target_loader = load_data(i, args)
    print("Data loaded.",flush=True)

    # create model
    predictor = Classifier(args.predictor_units, args.dropout, num_class=4)

    predictor.apply(weight_init)
    predictor.to(device)
    print("Model created.",flush=True)

    # create loss function and optimizer
    criterion_pred = nn.CrossEntropyLoss()
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)

    print("Start training.",flush=True)

    sp_best = 0.
    loss_list = []
    acc_list = []
    for epoch in range(args.epochs):
        loss_all = []
        predictor.train()

        for (src_data, src_label) in source_loader:
            src_data, src_label = src_data.to(device), src_label.to(device)
            optimizer_predictor.zero_grad()
            src_label_pred = predictor(src_data)
            
            loss= criterion_pred(src_label_pred, src_label)
            loss.backward()
            optimizer_predictor.step()
            loss_all.append(loss.item())
        loss_all = np.mean(loss_all)

        # evulate
        acc = evaluate(predictor, target_loader, device)

        loss_list.append(loss_all)
        acc_list.append(acc)

        print(f"Epoch {epoch}, train loss: {loss_all:.4f}, val acc: {acc:.4f}",flush=True)
        if acc > sp_best:
            sp_best = acc
            torch.save(predictor.state_dict(), save_path+f"predictor_ratio{args.ratio}")
            print("saving new best model",flush=True)
        # else:
        #     kill_cnt += 1
        #     if kill_cnt >= 3:
        #         print(f"early stop, best acc: {sp_best:.4f}")
        #     break


    # torch.save(discriminator, save_path+f"discriminator_ratio{args.ratio}")

    path = f'../log/mlp/{i}/'
    os.makedirs("../log", exist_ok=True)
    os.makedirs("../log/mlp", exist_ok=True)
    os.makedirs(f"../log/mlp/{i}", exist_ok=True)
    np.save(path+f'ratio{args.ratio}_loss.npy',np.array(loss_list))
    np.save(path+f'ratio{args.ratio}_acc.npy',np.array(acc_list))

    # Third, test step. Predict target
    print("Start testing.",flush=True)
    predictor.load_state_dict(torch.load(save_path+f"predictor_ratio{args.ratio}"))
    acc = evaluate(predictor, target_loader, device)
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

    os.makedirs("../ckpts", exist_ok=True)
    os.makedirs("../ckpts/mlp", exist_ok=True)
    acc = []
    for i in range(1, 16):
        os.makedirs(f"../ckpts/mlp/{i}", exist_ok=True)
        cur_acc = main(args, i, save_path=f"../ckpts/mlp/{i}/")
        acc.append(cur_acc)
    print(f"average acc: {np.mean(acc)}",flush=True)
