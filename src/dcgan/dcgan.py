import torch
import torch.nn as nn
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
nb_channel = 3
nb_gen_features = 64
nb_dis_features = 64

class Generator(nn.Module):
    def __init__(self, latent_dim=latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, nb_gen_features * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nb_gen_features * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nb_gen_features * 8, nb_gen_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_gen_features * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nb_gen_features * 4, nb_gen_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_gen_features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nb_gen_features * 2, nb_gen_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_gen_features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nb_gen_features, nb_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nb_channel, nb_dis_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_dis_features, nb_dis_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_dis_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_dis_features * 2, nb_dis_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_dis_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_dis_features * 4, nb_dis_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nb_dis_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_dis_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

class DCGAN:
    def __init__(self, latent_dim=latent_dim, lr=0.0002):
        self.latent_dim = latent_dim
        self.G = Generator(latent_dim).to(device)
        self.D = Discriminator().to(device)
        self.loss = nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    def show(self):
        print(self.G)
        print(self.D)

    def train(self, data, epochs):
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(data, 0):
                real_imgs = imgs.to(device)

                real_label = torch.full((imgs.size(0),), 1.0, dtype=torch.float, device=device)
                fake_label = torch.full((imgs.size(0),), 0.0, dtype=torch.float, device=device)

                # == Discriminator Train ==
                # Real data
                self.D.zero_grad()
                d_output = self.D(real_imgs).view(-1)
                real_loss = self.loss(d_output, real_label)
                real_loss.backward()
                self.optimizer_D.step()
                d_loss_real = d_output.mean().item()

                # Fake data
                noise = torch.randn(imgs.size(0), self.latent_dim, 1, 1, device=device)
                fake_imgs = self.G(noise)
                d_output = self.D(fake_imgs.detach()).view(-1)
                fake_loss = self.loss(d_output, fake_label)
                fake_loss.backward()
                d_loss_fake = d_output.mean().item()
                D_loss = real_loss + fake_loss
                self.optimizer_D.step()

                # == Generator Train ==
                self.G.zero_grad()
                g_output = self.D(fake_imgs).view(-1)
                g_loss = self.loss(g_output, real_label)
                g_loss.backward()
                G_loss = g_output.mean().item()
                self.optimizer_G.step()

                if i % 5 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, epochs, i, len(data),
                            D_loss.item(), g_loss.item(), d_loss_real, d_loss_fake, G_loss))

                if epoch % 5 == 0 and i == 0:
                    with torch.no_grad():
                        sample_imgs = self.G(self.fixed_noise)
                    save_image(sample_imgs.data[:25], "../img/dcgan_epoch_%d.png" % epoch, nrow=5, normalize=True)
