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
    source_encoder = MLP(args.encoder_units, args.dropout)
    target_encoder = MLP(args.encoder_units, args.dropout)
    predictor = Classifier(args.predictor_units, args.dropout, num_class=4)
    discriminator = Classifier(args.discriminator_units, args.dropout, num_class=2)


    source_encoder.apply(weight_init)
    predictor.apply(weight_init)
    discriminator.apply(weight_init)

    source_encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)
    discriminator.to(device)
    print("Model created.",flush=True)

    # create loss function and optimizer
    criterion_pred = nn.CrossEntropyLoss()
    criterion_discrim = nn.CrossEntropyLoss()
    optimizer_source = optim.Adam(
        list(source_encoder.parameters())+list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )
    optimizer_target = optim.Adam(target_encoder.parameters(), lr=1e-3, weight_decay=1e-2)
    optimizer_discrim = optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-2)

    # First, pretrain step. Train source_encoder + label_predictor
    print("Start training source_encoder and label_predictor.",flush=True)

    sp_best = 0.
    kill_cnt = 0
    for epoch in range(args.epochs):
        tr_loss = []
        source_encoder.train()
        predictor.train()

        # train
        for data, labels in source_loader:
            data, labels = data.to(device), labels.to(device)

            source_feat = source_encoder(data)
            label_pred = predictor(source_feat)
            loss = criterion_pred(label_pred, labels)

            optimizer_source.zero_grad()
            loss.backward()
            optimizer_source.step()

            tr_loss.append(loss.item())

        tr_loss = np.mean(tr_loss)

        # evulate
        acc = evaluate(nn.Sequential(source_encoder, predictor), target_loader, device)

        print(f"Epoch {epoch}, train loss: {tr_loss:.4f}, val acc: {acc:.4f}",flush=True)
        if acc > sp_best:
            sp_best = acc
            torch.save(source_encoder.state_dict(), save_path+"source_encoder")
            torch.save(predictor.state_dict(), save_path+"predictor")
            print("saving new best model",flush=True)
        # else:
        #     kill_cnt += 1
        #     if kill_cnt >= 3:
        #         print(f"early stop, best acc: {sp_best:.4f}")
        #     break

    # Second, adapt step. Train target_encoder + disriminator
    print("Start training target_encoder and discriminator.",flush=True)
    target_encoder.load_state_dict(torch.load(save_path+"source_encoder"))
    source_encoder.load_state_dict(torch.load(save_path+"source_encoder"))

    for epoch in range(args.epochs):
        loss_all = []
        target_encoder.train()
        discriminator.train()
        source_encoder.eval()

        for (tgt_data, _), (src_data, _) in zip(target_loader, source_loader):
            tgt_data, src_data = tgt_data.to(device), src_data.to(device)

            optimizer_discrim.zero_grad()
            optimizer_target.zero_grad()

            src_feat = source_encoder(src_data)
            tgt_feat = target_encoder(tgt_data)
            src_pred = discriminator(src_feat.detach())
            tgt_pred = discriminator(GRL.apply(tgt_feat, 2/(1+np.exp(-100*epoch/args.epochs))-1))

            src_labels = torch.ones(src_pred.shape).to(device)
            tgt_labels = torch.zeros(tgt_pred.shape).to(device)

            loss_total = criterion_discrim(torch.cat((src_pred, tgt_pred)), torch.cat((src_labels, tgt_labels)))
            loss_total.backward()

            optimizer_discrim.step()
            optimizer_target.step()

            loss_all.append(loss_total.item())
        loss_all = np.mean(loss_all)
        print(f"Epoch {epoch}, train loss: {loss_all:.4f}",flush=True)

    
    torch.save(discriminator, save_path+"discriminator")
    torch.save(target_encoder, save_path+"target_encoder")

    # Third, test step. Predict target
    print("Start testing.",flush=True)
    predictor.load_state_dict(torch.load(save_path+"predictor"))
    target_predictor = nn.Sequential(target_encoder, predictor)
    acc = evaluate(target_predictor, target_loader, device)
    print(f"Test on person {i}, acc: {acc:.4f}",flush=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder_units", type=int, nargs='+', default=[310, 256, 128])
    parser.add_argument("--predictor_units", type=int, nargs='*', default=[128, 64, 32])
    parser.add_argument("--discriminator_units", type=int, nargs='*', default=[128, 32])
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('-r',"--ratio", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs("../ckpts", exist_ok=True)
    os.makedirs("../ckpts/adda", exist_ok=True)

    acc = []
    for i in range(1, 16):
        os.makedirs(f"../ckpts/adda/{i}", exist_ok=True)
        cur_acc = main(args, i, save_path=f"../ckpts/adda/{i}/")
        acc.append(cur_acc)
    print(f"average acc: {np.mean(acc)}",flush=True)