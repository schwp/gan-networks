import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        z = img.view(img.size(0), -1)
        return self.model(z)

class GAN:
    def __init__(self, latent_dim, img_size, lr=0.0002):
        self.latent_dim = latent_dim
        self.G = Generator(latent_dim, img_size).to(device)
        self.D = Discriminator(img_size).to(device)
        self.loss = nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.Tensor = torch.FloatTensor

    def train(self, data, epochs):
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(data):

                valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
                real_imgs = Variable(imgs.type(self.Tensor))

                # == Generator Train ==
                self.optimizer_G.zero_grad()
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))
                gen_imgs = self.G(z)
                g_loss = self.loss(self.D(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # == Discriminator Train ==
                self.optimizer_D.zero_grad()
                real_loss = self.loss(self.D(real_imgs), valid)
                fake_loss = self.loss(self.D(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                if i % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, epochs, i, len(data), d_loss.item(), g_loss.item())
                    )

                if epoch % 5 == 0 and i == 0:
                    save_image(gen_imgs.data[:25], "img/epoch_%d.png" % epoch, nrow=5, normalize=True)
