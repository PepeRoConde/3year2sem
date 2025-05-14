import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from transformacions import PostProcesado

def grafica_prediccions(modelo, cargador, dispositivo, nome, filas=2 ,mostra=False, postprocesado=True):
    modelo.eval()
    fig, axs = plt.subplots(filas, 5 if postprocesado else 4, figsize=(20, 5 * filas))

    for i in range(filas):
        idx = np.random.randint(0, len(cargador.dataset) - 1)
        imaxe, gt = cargador.dataset[idx]

        imaxe = imaxe.to(dispositivo)
        gt = gt.to(dispositivo)

        with torch.no_grad(): prediccion = modelo(imaxe.unsqueeze(0))

        if imaxe.shape[0] > 1: imaxe_vis = imaxe[0].cpu().numpy()
            
        else: imaxe_vis = imaxe.squeeze().cpu().numpy()
            

        gt_np = gt.squeeze().cpu().numpy()
        pred_np = torch.sigmoid(prediccion).squeeze().cpu().numpy()
        pred_bin = (pred_np > 0.5).astype(np.uint8)

        overlay = np.stack([gt_np, pred_np, np.zeros_like(gt_np)], axis=-1)
        overlay = np.clip(overlay, 0, 1)

        j = 0
        axs[i, j].imshow(imaxe_vis, cmap='gray')
        axs[i, j].set_title(f"Orixinal {i + 1}")
        axs[i, j].axis('off')

        j += 1
        axs[i, j].imshow(gt_np, cmap='gray')
        axs[i, j].set_title(f"GT {i + 1}")
        axs[i, j].axis('off')

        j += 1
        axs[i, j].imshow(pred_np, cmap='gray')
        axs[i, j].set_title(f"Prediccion {i + 1}")
        axs[i, j].axis('off')

        if postprocesado:
            j += 1
            postprocesado = PostProcesado()
            pred_procesada = postprocesado(pred_bin)
            overlay = np.stack([gt_np, pred_procesada, np.zeros_like(gt_np)], axis=-1)
            overlay = np.clip(overlay, 0, 1)
            axs[i, j].imshow(pred_procesada, cmap='gray')
            axs[i, j].set_title(f"Apertura {i + 1}")
            axs[i, j].axis('off')

        j += 1
        axs[i, j].imshow(imaxe_vis, cmap='gray')
        axs[i, j].imshow(overlay, alpha=0.5)
        axs[i, j].set_title(f"GT e prediccions {i + 1}")
        axs[i, j].axis('off')

    plt.tight_layout()
    if mostra:
        plt.show()
    plt.savefig('../figuras/PR_'+nome+'.jpg')
    plt.close()



def grafica_canles(tensor, titulos=None):
    if tensor.ndimension() == 4: # por si é un batch
        tensor = tensor[0]
    
    C = tensor.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(15, 5))

    if titulos is None:
        titulos = [f"Canle {i + 1}" for i in range(C)]

    for i in range(C):
        ax = axes[i]
        channel_data = tensor[i].cpu().numpy()  # Convert tensor to numpy for plotting
        im = ax.imshow(channel_data, cmap='plasma')
        ax.set_title(titulos[i] if i < len(titulos) else f"Canle {i + 1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

def grafica_aumento_datos(ConxuntoDatos, n, m, ruta='../OCT-dataset'):
    datos_orixinais = ConxuntoDatos(ruta=ruta, aumento_datos=False)
    datos_aumentados = ConxuntoDatos(ruta=ruta, aumento_datos=True)
    fig, axes = plt.subplots(n, m * 2, figsize=(12, 6))

    for i in range(n * m):
        orixinal, _ = datos_orixinais[i]
        aumentada, _ = datos_aumentados[i]

        ax = axes[i // m, (i % m) * 2]
        ax.imshow(orixinal.permute(1, 2, 0).numpy(), cmap='plasma')
        ax.set_title(f"Orixinal {i+1}")
        ax.axis('off')

        ax = axes[i // m, (i % m) * 2 + 1]
        ax.imshow(aumentada.permute(1, 2, 0).numpy(), cmap='plasma')
        ax.set_title(f"Aumentada {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

def grafica_curvas(perdidas, metricas, nome, mostra):
    def to_numpy_if_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    perdidas = {k: to_numpy_if_tensor(v) for k, v in perdidas.items()}
    metricas = {k: to_numpy_if_tensor(v) for k, v in metricas.items()}

    epocas = range(1, len(next(iter(perdidas.values()))) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Perdidas
    for perdida, valores in perdidas.items():
        ax1.plot(epocas, valores, label=perdida)
    ax1.set_title('Perdidas')
    ax1.set_xlabel('Epoca')
    ax1.legend()
    ax1.grid(True)

    # Metricas
    for metrica, valores in metricas.items():
        ax2.plot(epocas, valores, label=metrica)
    ax2.set_title('Metricas')
    ax2.set_xlabel('Epoca')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if mostra:
        plt.show()
    plt.savefig('../figuras/CU_'+nome+'.jpg')
    plt.close()

