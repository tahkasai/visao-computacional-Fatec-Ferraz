import cv2

# Leitura da imagem com aspas
imOriginal = cv2.imread('04 operacao morfologica/imagem/teste.bmp', 0)

# Define o elemento estruturante
eEstruturante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Aplica o operador morfológico
imProc = cv2.erode(imOriginal, eEstruturante, iterations=2)

# Mostra as imagens com os nomes entre aspas
cv2.imshow('Imagem original', imOriginal)
cv2.imshow('Imagem processada', imProc)

# Espera pressionar qualquer tecla
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salva a imagem processada com nome correto, extensão e aspas
cv2.imwrite('04 operacao morfologica/imagem/imagem_processada.bmp', imProc)
