import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm, trange
import sys
sys.path.append("..")
from models.ADDA import MLP, Classifier
from utils import load_dg_data, seed_all, weight_init, GRL
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
    source_loader, target_loader = load_dg_data(i, args)
    print("Data loaded.",flush=True)

    # create model
    encoder = MLP(args.encoder_units, args.dropout)
    predictor = Classifier(args.predictor_units, args.dropout, num_class=4)
    discriminator = Classifier(args.discriminator_units, args.dropout, num_class=15)


    encoder.apply(weight_init)
    predictor.apply(weight_init)
    discriminator.apply(weight_init)

    encoder.to(device)
    predictor.to(device)
    discriminator.to(device)
    print("Model created.",flush=True)

    # create loss function and optimizer
    criterion_pred = nn.CrossEntropyLoss()
    criterion_discrim = nn.CrossEntropyLoss()

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_discrim = optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-2)

    print("Start training.",flush=True)

    sp_best = 0.
    kill_cnt = 0
    loss_list = []
    acc_list = []
    for epoch in range(args.epochs):
        loss_all = []
        encoder.train()
        discriminator.train()
        predictor.train()

        for (tgt_data, _), (src_data, src_label, src_dom_label) in zip(target_loader, source_loader):
            tgt_data, src_data, src_label = tgt_data.to(device), src_data.to(device), src_label.to(device)
            src_dom_label = src_dom_label.long().to(device)

            optimizer_discrim.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_predictor.zero_grad()

            src_feat = encoder(src_data)
            tgt_feat = encoder(tgt_data)

            src_dom_pred = discriminator(GRL.apply(src_feat, 2/(1+np.exp(-100*epoch/args.epochs))-1))
            tgt_dom_pred = discriminator(GRL.apply(tgt_feat, 2/(1+np.exp(-100*epoch/args.epochs))-1))
            src_label_pred = predictor(src_feat)

            tgt_dom_label = torch.full((tgt_dom_pred.shape[0],), 14, device=device)

            loss_discrim = criterion_discrim(torch.cat((src_dom_pred, tgt_dom_pred)), torch.cat((src_dom_label, tgt_dom_label)))
            loss_pred = criterion_pred(src_label_pred, src_label)

            loss_total = loss_discrim + loss_pred
            loss_total.backward()

            optimizer_encoder.step()
            optimizer_discrim.step()
            optimizer_predictor.step()

            loss_all.append(loss_total.item())
        loss_all = np.mean(loss_all)

        # evulate
        acc = evaluate(nn.Sequential(encoder, predictor), target_loader, device)

        loss_list.append(loss_all)
        acc_list.append(acc)

        print(f"Epoch {epoch}, train loss: {loss_all:.4f}, val acc: {acc:.4f}",flush=True)
        if acc > sp_best:
            sp_best = acc
            torch.save(encoder.state_dict(), save_path+f"encoder_ratio{args.ratio}")
            torch.save(predictor.state_dict(), save_path+f"predictor_ratio{args.ratio}")
            print("saving new best model",flush=True)
        # else:
        #     kill_cnt += 1
        #     if kill_cnt >= 3:
        #         print(f"early stop, best acc: {sp_best:.4f}")
        #     break


    torch.save(discriminator, save_path+f"discriminator_ratio{args.ratio}")

    path = f'../log/dgdann/{i}/'
    os.makedirs("../log", exist_ok=True)
    os.makedirs("../log/dgdann", exist_ok=True)
    os.makedirs(f"../log/dgdann/{i}", exist_ok=True)
    np.save(path+f'ratio{args.ratio}_loss.npy',np.array(loss_list))
    np.save(path+f'ratio{args.ratio}_acc.npy',np.array(acc_list))

    
    # Third, test step. Predict target
    print("Start testing.",flush=True)
    predictor.load_state_dict(torch.load(save_path+f"predictor_ratio{args.ratio}"))
    encoder.load_state_dict(torch.load(save_path+f"encoder_ratio{args.ratio}"))
    target_predictor = nn.Sequential(encoder, predictor)
    acc = evaluate(target_predictor, target_loader, device)
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
    parser.add_argument("--predictor_units", type=int, nargs='*', default=[128, 64, 32])
    parser.add_argument("--discriminator_units", type=int, nargs='*', default=[128, 48])
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('-r',"--ratio", type=float, default=1.0)
    parser.add_argument('-t',"--aug_type", type=int, default=4)
    parser.add_argument('-s',"--data_source", type=int, default=0)
    args = parser.parse_args()
    print(args)

    os.makedirs("../ckpts", exist_ok=True)
    os.makedirs("../ckpts/dg_dann", exist_ok=True)
    acc = []
    for i in range(1, 16):
        os.makedirs(f"../ckpts/dg_dann/{i}", exist_ok=True)
        cur_acc = main(args, i, save_path=f"../ckpts/dg_dann/{i}/")
        acc.append(cur_acc)
    print(f"average acc: {np.mean(acc)}",flush=True)
