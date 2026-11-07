import cv2
import numpy as np

caminho_imagem = "05 atividade/image/cupinzeiro.jpeg"
imagem = cv2.imread(caminho_imagem)

if imagem is None:
    raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

largura = 800
proporcao = largura / imagem.shape[1]
altura = int(imagem.shape[0] * proporcao)
imagem = cv2.resize(imagem, (largura, altura))

hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
desfoque = cv2.GaussianBlur(hsv, (5, 5), 0)

h_min = 6
s_min = 60
v_min = 40

h_max = 24
s_max = 240
v_max = 190

limite_inferior = np.array([h_min, s_min, v_min])
limite_superior = np.array([h_max, s_max, v_max])

mascara = cv2.inRange(desfoque, limite_inferior, limite_superior)

kernel_abertura = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mascara_limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_abertura)

tamanho_fechamento = 15
kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_fechamento, tamanho_fechamento))
mascara_final = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel_fechamento)

area_minima = 4000

contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagem_resultado = imagem.copy()

maior_contorno = None
area_maxima = 0

for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area > area_minima and area > area_maxima:
        maior_contorno = cnt
        area_maxima = area

if maior_contorno is not None:
    cv2.drawContours(imagem_resultado, [maior_contorno], -1, (0, 255, 0), 3)

    x, y, w, h = cv2.boundingRect(maior_contorno)
    cv2.putText(imagem_resultado, f"Area: {area_maxima:.0f} px", (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"Área detectada: {area_maxima:.2f} px²")
else:
    print("Nenhum contorno válido encontrado com a área mínima definida.")

cv2.imshow("Mascara (Preto e Branco)", mascara_final)
cv2.imshow("Resultado", imagem_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()