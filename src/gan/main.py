from gan import GAN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

latent_dim = 64
img_size = 28*28
batch_size = 32
lr = 0.0002
epochs = 50

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root="../../data/mnist", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    gan = GAN(latent_dim, img_size, lr)
    gan.train(dataloader, epochs)