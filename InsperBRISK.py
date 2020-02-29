# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:07:24 2020

@author: Usuario
"""

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Pegando imagem

img_total = cv2.imread('folha.png')
insper_logo = cv2.imread('folha_atividade.png')
insper_logo_gray = cv2.cvtColor(insper_logo, cv2.COLOR_BGR2GRAY)

# Identificando as features da imagem

brisk = cv2.BRISK_create()
kpts = brisk.detect(insper_logo_gray)

x = [k.pt[0] for k in kpts]
y = [k.pt[1] for k in kpts]

s = [(k.size/2)**2 * math.pi for k in kpts]

# Visualizando as imagens

plt.figure(figsize=(10, 10))
colors = list("bgrcmy")
cores_grafico = np.random.choice(colors, size=len(x), replace=True)
plt.scatter(x, y, s, c=cores_grafico, alpha=0.12)
plt.imshow(insper_logo_gray, cmap=cm.gray)
plt.title('BRISK feature')

# Essa função vai ser usada para encontrar a matriz que devolve a imagem com retangulo

def find_homography_draw_box(kp1, kp2, img_cena):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()


    
    h, w = img_total_gray.shape
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)


    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b

# Função find good matches acha matches
    
def find_good_matches(descriptor_image1, frame_gray):
    """
        Recebe o descritor da imagem a procurar e um frame da cena, e devolve os keypoints e os good matches
    """
    des1 = descriptor_image1
    kp2, des2 = brisk.detectAndCompute(frame_gray,None)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp2, good

# Carregar duas imagens e mostrar
    
MIN_MATCH_COUNT = 10 #Número mínimo de pontos correspondentes

insper_logo_rgb = cv2.cvtColor(insper_logo, cv2.COLOR_BGR2RGB)
img_total_rgb = cv2.cvtColor(img_total, cv2.COLOR_BGR2RGB)
img_total_gray = cv2.cvtColor(img_total, cv2.COLOR_BGR2GRAY)

framed = None

# Imagem de saída
out = img_total_rgb.copy()

# Cria o detector BRISK
brisk = cv2.BRISK_create()

# Encontra os pontos únicos (keypoints) nas duas imagems
kp1, des1 = brisk.detectAndCompute(insper_logo_gray, None)
kp2, des2 = brisk.detectAndCompute(img_total_gray, None)

# Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


# Tenta fazer a melhor comparacao usando o algoritmo
matches = bf.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    # Separa os bons matches na origem e no destino
    print("Matches found")    
    framed = find_homography_draw_box(kp1, kp2, img_total_rgb)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    
# Assim que o objeto for localizado, mostra a imagem final na tela

plt.figure(figsize=(10, 10))
plt.imshow(framed)

# DESENHANDO AS CORRESPONDÊNCIAS

img3 = cv2.drawMatches(insper_logo_rgb, kp1, img_total_rgb, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) #drawMatches desenha as features na tela

# Mostra na tela
 
plt.figure(figsize=(10, 10))
plt.imshow(img3)

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret == False:
        print("Problema para capturar o frame da câmera")
        continue
    
    frame_rgb = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kp2, good_matches = find_good_matches(des1, gray)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(img3, 'Press q to quit', (0, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    if len(good_matches) > MIN_MATCH_COUNT:
        img3 = cv2.drawMatches(insper_logo_rgb, kp1, frame_rgb, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('BRISK features', img3)
    else:
        cv2.imshow("BRISK features", frame)
        
    print(len(good_matches))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




