import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

batch_size = 128
lr = 5e-4
num_epochs = 1000

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(310, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,310)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(autoencoder,dataloader,device,num_epochs,model_idx):
    print(f"\n\nStarting Training model{model_idx}",flush=True)
    autoencoder.to(device)
    autoencoder.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            autoencoder.zero_grad()
            data = data.float().to(device)
            output = autoencoder(data)
            err = criterion(data,output)
            err.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_G: %.4f'
                    % (epoch+1, num_epochs, i, len(dataloader),err.item()),flush=True)

    os.makedirs('ae/',exist_ok=True)
    torch.save(autoencoder.state_dict(), f'ae/ae{model_idx}.pth')

def augmente(autoencoder,data,device,num_data):
    autoencoder.eval()
    autoencoder.to(device)
    idx = np.random.choice(np.arange(data.shape[0]),num_data)
    aug_data = autoencoder(data[idx])
    return aug_data


def AE_Aug(mode,data,num_data=100,model_idx=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        autoencoder = AutoEncoder()
        dataset = CustomDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(autoencoder,dataloader,device=device,num_epochs=num_epochs,model_idx=model_idx)
    
    else:
        autoencoder = AutoEncoder()
        if not os.path.exists(f'ae/ae{model_idx}.pth'):
            print("Error! No available model")
            exit(-1)
        autoencoder.load_state_dict(torch.load(f'ae/ae{model_idx}.pth'))
        autoencoder.eval()
        aug_data = augmente(autoencoder,data,device=device,num_data=num_data)
        print(f"Generate {aug_data.shape[0]} data")
        return aug_data.detach().to('cpu')