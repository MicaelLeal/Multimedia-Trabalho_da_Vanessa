# coding=utf-8
import numpy as np
import cv2
import mahotas


def escreve(img, texto, cor=(255,0,0)):
	fonte = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0, cv2.LINE_AA)


imgColorida = cv2.imread('dados.jpeg')

imgray = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)

suave = cv2.blur(imgray, (7, 7))

T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

bordas = cv2.Canny(bin, 70, 150)

contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

escreve(imgray, "Imagem em tons de cinza", 0)
escreve(suave, "Suavizacao com Blur", 0)
escreve(bin, "Binarizacao com Metodo Otsu", 255)
escreve(bordas, "Detector de bordas Canny", 255)

temp = np.vstack([
	np.hstack([imgray, suave]),
	np.hstack([bin, bordas])
])

cv2.imshow("Quantidade de objetos: "+str(len(contornos)), temp)
cv2.waitKey(0)

imgC2 = imgColorida.copy()
cv2.drawContours(imgC2, contornos, -1, (0,255,0), 3)
escreve(imgC2, str(len(contornos))+" objetos encontrados!")
cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)
