import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义超参数
batch_size = 128
lr = 0.0002
num_epochs = 1000

# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 310)
        )

    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(310, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train(netG,netD,dataloader,device,num_epochs):
    netG.to(device)
    netD.to(device)
    netG.train()

    criterionD = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real = data.float().to(device)
            b_size = real.size(0)
            label_real = torch.full((b_size,), 1.0, dtype=torch.float).to(device)
            label_fake = torch.full((b_size,), 0.0, dtype=torch.float).to(device)
            output = netD(real).squeeze()
            errD_real = criterionD(output, label_real)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, 100, dtype=torch.float).to(device)
            fake = netG(noise)
            output = netD(fake.detach()).squeeze()
            errD_fake = criterionD(output, label_fake)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            output = netD(fake).squeeze()
            D_G_z2 = output.mean().item()
            errG = criterionD(output, label_real)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2),flush=True)

        if epoch % 50 == 0:
            os.makedirs('dcgan/',exist_ok=True)
            torch.save(netG.state_dict(), f'dcgan/generator{epoch}.pth')
            torch.save(netD.state_dict(), f'dcgan/discriminator{epoch}.pth')

def augmente(netG,device,num_data):
    netG.eval()
    netG.to(device)
    noise = torch.randn(num_data, 100, dtype=torch.float).to(device)
    aug_data = netG(noise)
    return aug_data


def DCGAN_Aug(mode,data,num_data=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        netG = Generator()
        netD = Discriminator()
        dataset = CustomDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(netG,netD,dataloader,device=device,num_epochs=num_epochs)
    
    else:
        netG = Generator()
        if not os.path.exists('dcgan/generator.pth'):
            print("Error! No available model")
            exit(-1)
        netG.load_state_dict(torch.load('dcgan/generator400.pth'))
        aug_data = augmente(netG,device=device,num_data=num_data)
        print(f"Generate {aug_data.shape[0]} data")
        return aug_data.to('cpu')