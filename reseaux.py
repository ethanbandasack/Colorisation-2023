import numpy as np
import os
from PIL import Image
import shutil
import cv2 as cv
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = 'cuda'

class down(nn.Module):

    def __init__(self, in_features, out_features, stride = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, stride, 1)
        self.batch = nn.BatchNorm2d(out_features)
        self.activ = nn.LeakyReLU(.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        return x


class up(nn.Module):

    def __init__(self, in_features, out_features, activate=True, stride = True):
        super().__init__()
        if stride:
            self.conv = nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, 1)
        else:
            self.conv = nn.Conv2d(in_features, out_features, 3, 1, 1)
        if activate:
            self.activ = nn.LeakyReLU(.2)
            self.batch = nn.BatchNorm2d(out_features)
        else:
            self.activ = nn.Identity()
            self.batch = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        return x


class auto_encodeur(nn.Module):

    def __init__(self):
        super().__init__()
        self.down0 = down(3, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 256)
        self.up0 = up(256, 128)
        self.up1 = up(128, 64)
        self.up2 = up(64, 32)
        self.up3 = up(32, 16)
        self.up4 = up(16, 3)

    def forward(self, x):
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = nn.Tanh()(x)
        return x


def train(dataset, reseau, optimizer, epochs=5, loss = nn.MSELoss()):

    for _ in range(epochs):
        for i,x in tqdm(enumerate(DataLoader(dataset, 8, shuffle = True))):
            pages, pages_BW = x
            pages_color = reseau(pages_BW)
            calcul = loss(pages_color, pages)
            calcul.backward()
            optimizer.step()
            if i > 500:
                torch.save(reseau.state_dict(), "modèle")
                break
    for _,x in tqdm(enumerate(DataLoader(dataset, 8, shuffle = True))):
        pages, pages_BW = x
        pages_color = reseau(pages_BW)
        y = pages_color[0]
        y = y.view(y.shape[2], y.shape[1], y.shape[0])
        with torch.no_grad():
            plt.imshow(y*.5+.5)
        plt.show()
        break



class Disc(nn.Module):

    def __init__(self):
        super().__init__()
        self.down0 = down(3, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 128)
    
    def forward(self, x):
        
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class Gen(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.down0 = down(3, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 128)
        self.up0 = up(128, 64)
        self.up1 = up(64, 32)
        self.up2 = up(32, 16)
        self.up3 = up(16, 8)
        self.up4 = up(8, 3, False)
    
    def forward(self, x):
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = nn.Tanh()(x)
        return x

class Gen_Unet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.down0 = down(3, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 128)
        self.up0 = up(128, 64)
        self.up1 = up(64, 32)
        self.up2 = up(32, 16)
        self.up3 = up(16, 8)
        self.up4 = up(8, 3, False)
    
    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up0(x4)+x3
        x6 = self.up1(x5)+x2
        x7 = self.up2(x6)+x1
        x8 = self.up3(x7)+x0
        x9 = self.up4(x8)
        x10 = nn.Tanh()(x9)
        return x10

class down_RN(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.down1 = down(in_features, in_features, stride = 1)
        self.down2 = down(in_features, out_features)
        self.down3 = down(out_features, out_features, stride = 1)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.down2(x1)
        x1 = self.down3(x1)
        x2 = self.down2(x)
        return x1+x2


class up_RN(nn.Module):

    def __init__(self, in_features, out_features, activate=True):
        super().__init__()
        self.up1 = up(in_features, in_features, stride = False)
        self.up2 = up(in_features, out_features)
        self.up3 = up(out_features, out_features, activate, stride = False)
        self.up4 = up(in_features, out_features, activate)

    def forward(self, x):
        x1 = self.up1(x)
        x1 = self.up2(x1)
        x1 = self.up3(x1)
        x2 = self.up4(x)
        return x1+x2

class Disc_RN(nn.Module):

    def __init__(self):
        super().__init__()
        self.down0 = down_RN(3, 8)
        self.down1 = down_RN(8, 16)
        self.down2 = down_RN(16, 32)
        self.down3 = down_RN(32, 64)
        self.down4 = down_RN(64, 128)
    
    def forward(self, x):
        
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class Gen_RN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.down0 = down_RN(1, 8)
        self.down1 = down_RN(8, 16)
        self.down2 = down_RN(16, 32)
        self.down3 = down_RN(32, 64)
        self.down4 = down_RN(64, 128)
        self.up0 = up_RN(128, 64)
        self.up1 = up_RN(64, 32)
        self.up2 = up_RN(32, 16)
        self.up3 = up_RN(16, 8)
        self.up4 = up_RN(8, 3, False)
    
    
    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up0(x4)+x3
        x6 = self.up1(x5)+x2
        x7 = self.up2(x6)+x1
        x8 = self.up3(x7)+x0
        x9 = self.up4(x8)
        x10 = nn.Tanh()(x9)
        return x10

# gen = Gen_RN()
# from torchinfo import summary
# print(summary(gen, (128,3,256,256)))

def train_step(images, disc, gen, g_opt, d_opt, afficher=False, loss = nn.BCEWithLogitsLoss().to(device), imgloss = nn.MSELoss().to(device)):

    pages = images[0].to(device)
    pages_BW3 = images[1].to(device)
    pages_BW = pages_BW3[:,0:1].to(device)
    fake_images = gen(pages_BW)

    ### On entraine le discriminateur

    disc.zero_grad()

    fake_images_predictions = disc(fake_images.to(device))
    real_images_predictions = disc(pages).to(device)

    fake_labels = torch.zeros_like(fake_images_predictions)
    fake_labels = fake_labels.type_as(pages).to(device)
    real_labels = torch.ones_like(real_images_predictions)
    real_labels = real_labels.type_as(pages).to(device)

    ### On calcule la loss pour les vraies et les fausses images

    fake_loss = loss(fake_images_predictions, fake_labels).to(device)
    real_loss = loss(real_images_predictions, real_labels).to(device)

    disc_loss = (fake_loss + real_loss).to(device)

    ### Backpropagation

    disc_loss.backward()
    d_opt.step()
    #d_opt.zero_grad()



    ### On entraine le générateur

    gen.zero_grad()

    fake_images = gen(pages_BW).to(device)
    predictions_fake_images = disc(fake_images).to(device)

    if afficher:
        for i in range(len(fake_images)):
            fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(12,5))
            ax[0].set_title("Couleurs")
            ax[1].set_title("Colorisé")
            ax[2].set_title("Noir & blanc")
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[2].set_axis_off()
            img = pages_BW3[i]/2+.5
            ax[2].imshow(img.permute(2, 1, 0).cpu().detach())
            img = fake_images[i]/2+.5
            ax[1].imshow(img.permute(2, 1, 0).cpu().detach())
            img = pages[i]/2+.5
            ax[0].imshow(img.permute(2, 1, 0).cpu().detach())
            fig.savefig(f"img {i}.png")
            plt.close()

    real_labels = torch.ones_like(predictions_fake_images)
    real_labels = real_labels.type_as(pages).to(device)

    ### On calcule la loss pour le générateur

    gen_loss1 = loss(predictions_fake_images, real_labels).to(device)
    gen_loss2 = imgloss(fake_images, pages).to(device)

    gen_loss = (gen_loss1 + 100*gen_loss2).to(device)
    ### Backpropagation

    gen_loss.backward()
    g_opt.step()
    #g_opt.zero_grad()
    return gen_loss1, gen_loss2, disc_loss


def train(dataset,generator,discriminator,epochs=1,fixed_noise=torch.randn(size=(100, 3, 256, 256)).to('cuda')):

    loss = nn.BCELoss().to(device)
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
            gen_loss, disc_loss, img = train_step(image_batch, generator, discriminator, g_opt, d_opt)

            X.append(j)
            Lgen_loss.append(gen_loss.item())
            Ldisc_loss.append(disc_loss.item())
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}")

            # if i > 10:
            #     img = img/2+.5
            #     plt.imshow(img.permute(2, 1, 0).cpu().detach())
            #     plt.show()
            #     break

    #clear_output(wait=False)
    #generate_and_save_plots(X, Lgen_loss, Ldisc_loss) #Définie après, pour générer les courbes des loss
    #summarize_performance(generator,fixed_noise) #Définie après, pour afficher les images générées