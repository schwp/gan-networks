from dcgan import DCGAN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

img_size = 64
batch_size = 128

dataset = datasets.ImageFolder("../../data/celeba", 
                               transform=transforms.Compose([
                                   transforms.Resize((img_size, img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dcgan = DCGAN()
    # dcgan.show()
    dcgan.train(dataloader, epochs=50)
