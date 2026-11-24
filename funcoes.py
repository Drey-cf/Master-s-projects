from IPython.display import display
import av
import json
import io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import crop
from scipy import ndimage, datasets
from scipy.ndimage import uniform_filter, gaussian_filter
from tqdm import tqdm
import yuvio
import warnings
warnings.filterwarnings('ignore')

def jpeg_compress(original, quality = 75):
    buffer = io.BytesIO()
    Image.fromarray(original, 'L').save(buffer, format = "JPEG", quality = quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def MSE(original, pred):
    mse = np.mean(np.square(original.astype(np.float64) - pred.astype(np.float64)))
    return mse

def ssim_luc(original, rec, filtro, win_size=7, data_range=255, k1=0.01, k2=0.03, mse_g = np.nan):
    '''
    Objective: Calculates the original SSIM, with the Gaussian and Uniform filters.
    To use the Gaussian, only need the radius, and sigma. To use the uniform filter, only needs the window size.
    Inputs necessary: Original and reconstructed images and the filter used.
    Returns: the mean of the SSIM for the whole image. a float value'''
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    original = original.astype(np.float64, copy=False)
    rec = rec.astype(np.float64, copy=False)

    NP = win_size * win_size
    cov_norm = NP / (NP - 1)

    if filtro == 'uniform':
        # Local means
        ux = uniform_filter(original, size=win_size)
        uy = uniform_filter(rec, size=win_size)

        # Local second moments
        uxx = uniform_filter(original * original, size=win_size)
        uyy = uniform_filter(rec * rec, size=win_size)
        uxy = uniform_filter(original * rec, size=win_size)

    elif filtro == 'gaussian':
        # Local means
        ux = gaussian_filter(original, radius=win_size, sigma = 1.5)
        uy = gaussian_filter(rec, radius=win_size, sigma = 1.5)

        # Local second moments
        uxx = gaussian_filter(original * original, radius=win_size, sigma = 1.5)
        uyy = gaussian_filter(rec * rec, radius=win_size, sigma = 1.5)
        uxy = gaussian_filter(original * rec, radius=win_size, sigma = 1.5)

    # Local variances and covariance
    vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)
    vy = np.maximum(cov_norm * (uyy - uy * uy), 0.0)
    vxy = cov_norm * (uxy - ux * uy)

    # SSIM formula
    A1 = 2 * ux * uy + c1
    A2 = 2 * vxy + c2
    B1 = ux**2 + uy**2 + c1
    B2 = vx + vy + c2

    # Local SSIM
    S = (A1 * A2) / (B1 * B2)

    pad = (win_size - 1) // 2
    # Global SSIM (excluding borders)
    return crop(S, pad).mean(dtype=np.float64)

def ssim_martini(original, rec, filtro, win_size=7, data_range=255, k1=0.01, k2=0.03, mse_g = np.nan):
    '''
    Objective: Calculates the estimated SSIM, with the Martini estimation.
    Returns: the mean of the SSIM for the whole image. A float value'''
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    original = original.astype(np.float64, copy=False)
    rec = rec.astype(np.float64, copy=False)

    NP = win_size * win_size
    cov_norm = NP / (NP - 1)

    # Local means
    ux = uniform_filter(original, size=win_size)
    uy = uniform_filter(rec, size=win_size)

    # Local second moments
    uxx = uniform_filter(original * original, size=win_size)
    
    # Local variances and covariance
    vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)

    mse_l =  np.maximum(uniform_filter((original - rec) ** 2, size=win_size), 0.0)

    S = 1 - mse_l/ (2*vx + c2)

    pad = (win_size - 1) // 2

    return crop(S, pad).mean(dtype=np.float64)

def mssim_global(original, mse_g, rec = 0, win_size = 7, data_range=255, k1=.01, k2=.03, filtro = 'uniform'):
    '''
    Objective: Calculate the SSIM based on the global MSE of the original and reconstructed images.
    Inputs necessary: Original image and Global MSE.
    Returns: the mean of the SSIM for the whole image.'''
    c2 = (k2*data_range)**2

    original = original.astype(np.float64, copy=False)

    NP = win_size * win_size
    cov_norm = NP / (NP - 1)

    if filtro == 'uniform':
        ux = ndimage.uniform_filter(original, size=win_size)
        uxx = ndimage.uniform_filter(original * original, size=win_size)
        vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)

    elif filtro == 'gaussian':
        ux = ndimage.uniform_filter(original, size=win_size)
        uxx = ndimage.uniform_filter(original * original, size=win_size)
        vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)

    pad = (win_size - 1) // 2
    vx_c = crop(vx, pad)

    mssim = np.mean(1 - mse_g / (2*vx_c + c2))   # média de razões (a sua, sem bordas)

    return mssim

def calcular_mssim_por_taxa_por_imagem(funcao_ssim, nome, rec = np.nan, filtro = 'uniforme', dataset = 'kodak'):
    '''
    Objective: calculates the SSIM following a given function to it. But for each compress tax in arange(30, 100, 5).
    Input necessary: the SSIM function, the name of the returned dataframe
    Returns: 2 dataframes, The first contains the mssim for each quality tax, of the whole dataset.
    the second contains the ssims per image and for each compress tax.'''
    qualities = np.arange(30, 99, 5)

    ssim_dict_por_taxa_por_imagem = {}

    for quality in tqdm(qualities):

        ssim_lista = []

        if dataset == 'kodak':
            for image_name in kodak_dataset:

                path_image = path_kodak_ds + '/' + image_name

                img_original = Image.open(path_image).convert('L') # L de Luminancia L = R * 299/1000 + G * 587/1000 + B * 114/1000
                img_original = np.array(img_original, dtype = np.uint8)

                img_rec = jpeg_compress(img_original, quality = int(quality))

                mse_g = MSE(img_original, img_rec)

                mssim_de_uma_imagem = funcao_ssim(original = img_original, mse_g = mse_g, rec = img_rec, filtro = filtro)

                ssim_lista.append(mssim_de_uma_imagem)

            ssim_dict_por_taxa_por_imagem[int(quality)] = ssim_lista

            ssim_df_por_taxa_por_imagem = pd.DataFrame(ssim_dict_por_taxa_por_imagem, index = [n.replace('.png', '') for n in kodak_dataset])
            ssim_df_por_taxa_por_imagem.index = ['kodim' + str(i) for i in range(24)]

        elif dataset == 'subset':

            for img in subset_dataset:
                frame_.clear()

                container = av.open(path_subset + img)

                for i, frame in enumerate(container.decode(video=0)):
                    # Convert directly to a PIL.Image.Image (RGB)
                    Y = frame.to_image().convert('L')

                img_original = Y
                img_original = np.array(img_original, dtype = np.uint8)

                img_rec = jpeg_compress(img_original, quality = int(quality))
                mse_g = MSE(img_original, img_rec)

                mssim_de_uma_imagem = funcao_ssim(original = img_original, mse_g = mse_g, rec = img_rec, filtro = filtro)

                ssim_lista.append(mssim_de_uma_imagem)

            ssim_dict_por_taxa_por_imagem[int(quality)] = ssim_lista

            ssim_df_por_taxa_por_imagem = pd.DataFrame(ssim_dict_por_taxa_por_imagem, index = [n.replace('.y4m', '') for n in subset_dataset])
            ssim_df_por_taxa_por_imagem.index = ['subset' + str(i) for i in range(len(subset_dataset))]


    return ssim_df_por_taxa_por_imagem

def ssim_beta(original, mse_g, rec = 0, win_size=7, data_range = 255, k1=.01, k2=.03, filtro = ''):
    c2 = (k2*data_range)**2

    original = original.astype(np.float64, copy=False)

    NP = win_size * win_size
    cov_norm = NP / (NP - 1)

    ux = ndimage.uniform_filter(original, size=win_size)
    uxx = ndimage.uniform_filter(original * original, size=win_size)
    vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)

    pad = (win_size - 1) // 2
    vx_c = crop(vx, pad)

    epsilon = 1e-6
    vx_epsilon_beta = (vx_c + epsilon)**beta
    valor_esperado = np.mean(vx_epsilon_beta)
    mse_l = (mse_g/valor_esperado)*vx_epsilon_beta

    mssim = np.mean(1 - mse_l / (2*vx_c + c2))   # média de razões (a sua, sem bordas)

    return mssim

# Define a função comparando_implementacoes
qualities = np.arange(30, 99, 5)

def comparando_implementacoes(medias, labels, cores, erros, titulo, xticks = [i for i in range(30, 101, 10)], yticks = [0.70, 0.80, 0.90, 1.00], loc_legend = 'best'):
    fig, ax = plt.subplots(figsize = (5, 5))

    if erros != 0:
        for media, erro, label, cor in zip (medias, erros, labels, cores):
            ax.errorbar(qualities, media, erro, fmt = '-o', label = label, color = cor, markersize = 5)
    else:
        for media, label, cor in zip (medias, labels, cores):
            ax.plot(qualities, media, '-o', label = label, color = cor, markersize = 4)
    
    ax.legend(loc = loc_legend)
    ax.grid()
    
    ax.set_title(titulo, loc = 'left')
    ax.set_xlabel('JPEG Quality')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_ylabel('MSSIM')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim([min(yticks), max(yticks)+.01])
    ax.set_xlim([min(xticks)-5, max(xticks)+10])

    return ax, fig
