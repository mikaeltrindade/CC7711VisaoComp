# Importando bibliotecas necessárias
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função para carregar a imagem e converter para o espaço de cores RGB
def carregar_imagem(nome_arquivo):
    img = cv2.imread(nome_arquivo)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Convertendo a imagem para escala de cinza
def converter_para_escala_de_cinza(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

# Aplicando threshold para segmentação de objetos na imagem
def aplicar_threshold(img_gray):
    valor_maximo = img_gray.max()
    _, thresh = cv2.threshold(img_gray, valor_maximo / 15, valor_maximo, cv2.THRESH_BINARY_INV)
    return thresh

# Aplicando operações morfológicas para melhorar a segmentação
def aplicar_operacoes_morfologicas(thresh):
    tamanho_kernel = 10
    kernel = np.ones((tamanho_kernel, tamanho_kernel), np.uint8)
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh_open

# Aplicando filtro de borramento para reduzir ruídos
def aplicar_borramento(img_gray):
    tamanho_kernel = 10
    img_blur = cv2.blur(img_gray, ksize=(tamanho_kernel, tamanho_kernel))
    return img_blur

# Detectando bordas na imagem usando o algoritmo de Canny
def detectar_bordas(img_gray, threshold1, threshold2):
    edges = cv2.Canny(image=img_gray, threshold1=threshold1, threshold2=threshold2)
    return edges

# Encontrando contornos na imagem
def encontrar_contornos(edges_gray):
    contours, hierarchy = cv2.findContours(
                                   image=edges_gray,
                                   mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

# Desenhando contornos na imagem original
def desenhar_contornos(img, contours):
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
    return final

# Função para exibir as imagens
def exibir_imagens(imagens):
    formatoX = math.ceil(len(imagens) ** 0.5)
    if (formatoX ** 2 - len(imagens)) > formatoX:
        formatoY = formatoX - 1
    else:
        formatoY = formatoX
    for i in range(len(imagens)):
        plt.subplot(formatoY, formatoX, i + 1)
        plt.imshow(imagens[i], 'gray')
        plt.xticks([]), plt.yticks([])
    plt.show()

# Carregando a imagem
img = carregar_imagem('images/Aviao.jpeg')

# Convertendo a imagem para escala de cinza
img_gray = converter_para_escala_de_cinza(img)

# Aplicando threshold na imagem
thresh = aplicar_threshold(img_gray)

# Aplicando operações morfológicas na imagem thresholded
thresh_open = aplicar_operacoes_morfologicas(thresh)

# Aplicando filtro de borramento na imagem original
img_blur = aplicar_borramento(img_gray)

# Detectando bordas na imagem em escala de cinza e na imagem borradada
edges_gray = detectar_bordas(img_gray, threshold1=150, threshold2=200)
edges_blur = detectar_bordas(img_blur, threshold1=100, threshold2=200)

# Encontrando contornos na imagem thresholded
contours = encontrar_contornos(edges_gray)

# Desenhando contornos na imagem original
final = desenhar_contornos(img, contours)

# Exibindo as imagens
imagens = [img, img_blur, img_gray, edges_gray, edges_blur, thresh, thresh_open, final]
exibir_imagens(imagens)

# Exibindo apenas a imagem final com contornos
exibir_imagens([final])
