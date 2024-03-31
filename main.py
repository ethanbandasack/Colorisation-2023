import numpy as np
import os
from PIL import Image
import shutil
import cv2 as cv
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from reseaux import *
import wandb

path = "D://Ethan/Automatants/dataset/train/"
device = "cuda"

def gray(img):
    return .2989*img[:, :, 0]+.587*img[:, :, 1]+.114*img[:, :, 2]


def contours(img):
    SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    background = cv.morphologyEx(img, cv.MORPH_DILATE, SE)
    image = cv.divide(img, background, scale=255)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# L = os.listdir("./dataset/train")
# P = {t: None for t in L}

# for tome in tqdm(L):
#     pages = os.listdir(f"./dataset/train/{tome}/pages")
#     P[tome] = pages
#     try:
#         os.makedirs(f"./dataset/train/{tome}/pages_BW")
#     except:
#         pass

#     for page in pages:

#         planche = cv.imread(f"./dataset/train/{tome}/pages/{page}")
#         BW = contours(planche)
#         page_name = page.split('.')
#         page_BW = page_name[0]+'_BW.'+page_name[1]
#         (Image.fromarray(BW)).save(f"./dataset/train/{tome}/pages_BW/{page_BW}")


# with open('./index.csv', 'w') as f:
    # writer = csv.writer(f)
    # for t in tqdm(P):
    #     for page in P[t]: 
    #         page_name = page.split('.')
    #         page_BW = page_name[0]+'_BW.'+page_name[1]
    #         writer.writerow([path + t + '/pages/' + page,
    #                         path + t + '/pages_BW/' + page_BW])


class dataset(Dataset):

    def __init__(self, index_file):
        self.name = pd.read_csv(index_file)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, name):
        image = read_image(
            self.name.iloc[name, 0], ImageReadMode.RGB)/255*2-1
        image_NB = read_image(
            self.name.iloc[name, 1], ImageReadMode.RGB)/255*2-1
        return image, image_NB


# R = reseau1()
d = dataset("index.csv")
D = DataLoader(d, 32, shuffle = True)
# dataloader = DataLoader(d, 8, shuffle = True)
# for _, x in iter(dataloader):
#     break
# y = R(x)
# y = y[0]
# y = y.view(y.shape[2], y.shape[1], y.shape[0])
# with torch.no_grad():
#     plt.imshow(y*.5+.5)
# plt.show()

# R.load_state_dict(torch.load("modèle"))
# opt = torch.optim.Adam(R.parameters(),lr = 0.0001, betas = (0.5, 0.5))
# torch.save(R.state_dict(), "modèle")
# train(d, R, opt)
# torch.save(R.state_dict(), "modèle")


disc = Disc_RN()
gen = Gen_RN()
g_opt = gen.parameters()
d_opt = disc.parameters()

gen.load_state_dict(torch.load("Générateur_ResNet"))
disc.load_state_dict(torch.load("Discriminateur_ResNet"))

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "GAN",
    "dataset": "tkt",
    "epochs": 10,
    }
)
py 
# simulate training
epochs = 10
afficher = False
offset = 0

dataset=D
generator=gen
discriminator=disc
fixed_noise=torch.randn(size=(100, 3, 256, 256)).to('cuda')


generator = generator.to(device)
discriminator = discriminator.to(device)
g_opt = torch.optim.Adam(generator.parameters(),lr = 0.0001, betas = (0.5, 0.5))
d_opt = torch.optim.Adam(discriminator.parameters(),lr = 0.0001, betas = (0.5, 0.5))

Lgen_loss = []
Ldisc_loss = []
X = []
j = 0

for epoch in range(epochs):
    progress_bar = tqdm(dataset)
    ##Vu que c'est un dataloader, on ne peut itérer directement dessus avec son indice. On va juste prendre à chaque fois batch par batch.
    for i, image_batch in enumerate(progress_bar):
        j += 1
        page, page_couleur = image_batch
        real_images = page.to(device)
        gen_loss1, gen_loss2, disc_loss = train_step(image_batch, discriminator, generator, g_opt, d_opt, afficher)

        X.append(j)
        Lgen_loss.append((gen_loss1+gen_loss2).item())
        Ldisc_loss.append(disc_loss.item())
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss1+gen_loss2} | Disc Loss: {disc_loss}")

        # if i>10:
        #     break
    
        # log metrics to wandb
        wandb.log({"gen loss MSE": gen_loss2, "gen loss BCE": gen_loss1, "disc loss BCE": disc_loss})

    torch.save(generator.state_dict(), "Générateur_ResNet")
    torch.save(discriminator.state_dict(), "Discriminateur_ResNet")

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# print(train(D, disc, gen))