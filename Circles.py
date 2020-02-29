# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:04:17 2020

@author: Usuario
"""

import cv2
import numpy as np
import math

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

magenta_hsv1 = np.array([168,  50,  50], dtype=np.uint8)
magenta_hsv2 = np.array([178, 255, 255], dtype=np.uint8)

blue_hsv1 = np.array([84, 50, 50], dtype=np.uint8)
blue_hsv2 = np.array([150, 255, 255], dtype=np.uint8)

f=513
H=14

font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    blur = cv2.GaussianBlur(rgb,(5,5),0)
    
    bordas = auto_canny(blur)
    
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    circles = []
    
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
        
    #Linha de um centro pro outro
    
    if circles is not None:
        circles=circles[0]
        if len(circles)>=2:
            cv2.line(bordas_color, (circles[0][0], circles[0][1]), (circles[1][0], circles[1][1]), (0, 255, 255), 3)
            
            # Ângulo entre horizontal e linha entre circulos
            ang = math.atan2(circles[0][1]-circles[1][1], circles[0][0]-circles[1][0])
            angle = math.degrees(ang)
            
            cv2.putText(bordas_color, 'Angulo: {}'.format(angle), (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Distancia da tela
            dx=circles[0][0]-circles[1][0]
            dx_abs=abs(dx)
            dy=circles[0][1]-circles[1][1]
            dy_abs=abs(dy)
            dis=math.sqrt((dx*dx)+(dy*dy))
            D = (H*f)/(abs(dis))
            
            cv2.putText(bordas_color, 'Distancia: {}'.format(D), (0, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        else:
            cv2.putText(bordas_color, 'Angulo: None', (0, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bordas_color, 'Distancia: None', (0, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
        
    mask_magenta = cv2.inRange(hsv, magenta_hsv1, magenta_hsv2)
    
    mask_blue = cv2.inRange(hsv, blue_hsv1, blue_hsv2)
    
    bordas_color[mask_blue == 255] = [0, 0, 255]
    
    bordas_color[mask_magenta == 255] = [255, 0, 255]
    
    #cv2.imshow('Detector de circulos', mask_magenta)
    
    cv2.putText(bordas_color, 'Press Q to Quit', (0,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    bordas_color = cv2.cvtColor(bordas_color, cv2.COLOR_BGR2RGB)
    cv2.imshow('Detector de circulos', bordas_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()