import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Tamanho da imagem do dataset MNIST
img_shape = (1, 28, 28)

# Criaço do DataLoader
dataloader = DataLoader(
    FashionMNIST('.', download=True, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size = 128,
    shuffle = True
)


def bloco_gen_simples(dim_entrada, dim_saida, normalizar=True):
    """
    Função responsável por construir uma camada da rede Geradora.
    """
    camadas = [nn.Linear(dim_entrada, dim_saida)]
    if normalizar:
        camadas.append(nn.BatchNorm1d(dim_saida))
    camadas.append(nn.LeakyReLU(0.2, inplace=True))
    return camadas


class Gerador(nn.Module):
    """
    Classe que representa a rede Geradora.
    """

    def __init__(self, dim_ruido=64, dim_img=784, dim_oculta=128):
        """
        Função de criação da rede.
        """
        super(Gerador, self).__init__()

        self.modelo = nn.Sequential(
            *bloco_gen_simples(dim_ruido, dim_oculta * 2, False),
            *bloco_gen_simples(dim_oculta * 2, dim_oculta * 4),
            *bloco_gen_simples(dim_oculta * 4, dim_oculta * 8),
            nn.Linear(dim_oculta * 8, dim_img),
            nn.Sigmoid()
        )

    def forward(self, ruido):
        """
        Função responsável por fazer uma passagem do ruído por
        toda a estrutura da rede gerando uma imagem na saída.
        """
        return self.modelo(ruido)


def bloco_disc_simples(dim_entrada, dim_saida, normalizar=True):
    """
    Função responsável por construir uma camada da rede Discriminadora.
    """
    camadas = [nn.Linear(dim_entrada, dim_saida),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Dropout(0.3)]
    return camadas


class Discriminador(nn.Module):
    """
    Classe que representa a rede Discriminadora.
    """

    def __init__(self, dim_img=784, dim_oculta=128):
        """
        Função de criação da rede
        """
        super(Discriminador, self).__init__()

        self.modelo = nn.Sequential(
            *bloco_disc_simples(dim_img, dim_oculta * 4),
            *bloco_disc_simples(dim_oculta * 4, dim_oculta * 2),
            *bloco_disc_simples(dim_oculta * 2, dim_oculta),
            nn.Linear(dim_oculta, 1)
        )

    def forward(self, img):
        """
        Função responsável por fazer uma passagem da imagem pela rede
        tendo como saída predição. (0 para imagens falsas e 1 para
        imagens reais).
        """
        return self.modelo(img)


criterio = torch.nn.BCEWithLogitsLoss()
device = 'cpu'


def gerador_ruido(num_amostras, dim_ruido):
    """
    Função utilizada para gerar um ruído aleatório.
    """
    return torch.randn((num_amostras, dim_ruido), device=device)


def calc_perda_gen(discriminador, gerador, criterio, tam_batch):
    """
    Função que calcula a perda da rede Geradora.
    """
    # geração do ruído
    ruido = gerador_ruido(tam_batch, 64)

    # geração das imagens falsas
    imgs_falsas = gerador(ruido)

    # classificação do Discriminador para as imagens falsas
    disc_predicao_falsas = discriminador(imgs_falsas)

    # cálculo da perda
    g_perda = criterio(disc_predicao_falsas, torch.ones_like(disc_predicao_falsas, device=device))

    return g_perda


def calc_perda_disc(discriminador, gerador, criterio, tam_batch, imgs_real):
    """
    Função que calcula a perda da rede Discriminadora.
    """
    # geração do ruído
    ruido = gerador_ruido(tam_batch, 64)

    # geraço das imagens falsas
    imgs_falsas = gerador(ruido)

    # Predição para as imagens falsas e cálculo da perda 1
    disc_predicao_falsas = discriminador(imgs_falsas.detach())
    falsas_perda = criterio(disc_predicao_falsas, torch.zeros_like(disc_predicao_falsas, device=device))

    # Predição para as imagens reais e cálculo da perda 2
    disc_predicao_real = discriminador(imgs_real)
    real_perda = criterio(disc_predicao_real, torch.ones_like(disc_predicao_real, device=device))

    return (real_perda + falsas_perda) / 2


gerador = Gerador()
discriminador = Discriminador()


def exibir_imgs(imgs, num_imagens=25, dims=(1, 28, 28)):
    """
    Função para visualização de imagens. Cria um grid 5 por 5
    para criar imagens.
    """
    imgs_flat = imgs.detach().cpu().view(-1, *dims)
    imgs_grid = make_grid(imgs_flat[:num_imagens], nrow=5)
    plt.imshow(imgs_grid.permute(1, 2, 0).squeeze())
    plt.show()


media_disc_perdas = []
media_gen_perdas = []

for epoca in tqdm(range(100)):
    for img_real, _ in dataloader:
        # Pegando o tamanho do batch
        tam_batch = len(img_real)
        imgs_real = img_real.view(tam_batch, -1).to(device)

        otimizador_DISC.zero_grad()
        d_perda = calc_perda_disc(discriminador, gerador, criterio, tam_batch, imgs_real)
        d_perda.backward(retain_graph=True)
        otimizador_DISC.step()

        otimizador_GEN.zero_grad()
        g_perda = calc_perda_gen(discriminador, gerador, criterio, tam_batch)
        g_perda.backward(retain_graph=True)
        otimizador_GEN.step()


        media_disc_perdas.append(d_perda.item())
        media_gen_perdas.append(g_perda.item())

    if epoca % 10 == 0 and epoca != 0:
        ruido = gerador_ruido(tam_batch, 64)
        print(f'Epoca {epoca} | Gen Perda: {np.mean(media_gen_perdas)} | Disc Perda: {np.mean(media_disc_perdas)}')

        falsas_imgs = gerador(ruido)

        # Exibição das imagens geradas
        exibir_imgs(falsas_imgs)

        media_disc_perdas = []
        media_gen_perdas = []
