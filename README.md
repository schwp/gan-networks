# GAN Networks

## Overview
Implementation of GAN Neural Netwok using PyTorch.

A Vanilla GAN and DCGAN model are implemented and can be launched. The table down
below show which dataset the models use for training. You can see good the model
is doing with the images saved in `src/img/` through your model training.

| Model | Dataset (name & link)                                       | Output                           |
| ----- | ----------------------------------------------------------- | -------------------------------- |
| GAN   | MNIST (PyTorch download)                                    | Handwritten number image (28x28) |
| DCGAN | [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Celebrity face image (64x64)     |

## Run programs
In order to run the training of our network, just type the following command in your terminal and you will see logs about how the training is going :
```bash
cd src/{model_chosen}
python3 main.py
```
