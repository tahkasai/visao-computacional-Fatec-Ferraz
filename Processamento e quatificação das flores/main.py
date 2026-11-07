import cv2
import numpy as np
import os

def processar_imagem_flor(imagem, lower_color, upper_color):
    # Converte para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Segmentação por cor
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Limpeza da máscara (operações morfológicas)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Aplica a máscara na imagem original
    resultado = cv2.bitwise_and(imagem, imagem, mask=mask)
    
    # Cálculos
    total_pixels = imagem.shape[0] * imagem.shape[1]
    pixels_flor = cv2.countNonZero(mask)
    percentual_ocupacao = (pixels_flor / total_pixels) * 100 if total_pixels > 0 else 0
    
    return mask, resultado, total_pixels, pixels_flor, percentual_ocupacao

# Intervalo de cor para o tom roxo/azul das flores fornecidas
lower_blue_purple = np.array([110, 40, 40])
upper_blue_purple = np.array([165, 255, 255])

imagens_originais = ["flor1.jpeg", "flor2.jpeg"]

print(f"Diretório atual de execução: {os.getcwd()}\n")
print("--- PROCESSANDO FLORES REAIS ---")

for path in imagens_originais:
    if not os.path.exists(path):
        print(f"[ERRO] A imagem '{path}' não foi encontrada!")
        print(f"Verifique se o arquivo está salvo exatamente nesta pasta: {os.getcwd()}\n")
        continue
        
    img = cv2.imread(path)
    mask, resultado, total, px_flor, perc = processar_imagem_flor(img, lower_blue_purple, upper_blue_purple)
    
    print(f"Imagem: {path}")
    print(f"Total de pixels: {total}")
    print(f"Pixels correspondentes à flor: {px_flor}")
    print(f"Percentual de ocupação: {perc:.2f}%")
    
    # Salva os resultados para visualização
    cv2.imwrite(f"MASCARA_{path}", mask)
    cv2.imwrite(f"RESULTADO_{path}", resultado)
    print(f"-> Arquivos gerados com sucesso: 'MASCARA_{path}' e 'RESULTADO_{path}'\n")

print("--- PROCESSANDO IMAGEM SINTÉTICA (VALIDAÇÃO) ---")
# Gera uma imagem sintética simples (retângulo) para validar o algoritmo
img_sintetica = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.rectangle(img_sintetica, (100, 100), (300, 300), (200, 50, 150), -1) # Retângulo roxo
area_real_sintetica = 200 * 200 # Base x Altura (200x200 pixels)

mask_sint, res_sint, tot_sint, px_sint, perc_sint = processar_imagem_flor(img_sintetica, lower_blue_purple, upper_blue_purple)

if area_real_sintetica > 0:
    erro_percentual = (abs(area_real_sintetica - px_sint) / area_real_sintetica) * 100
else:
    erro_percentual = 0

print("Imagem Sintética 1 (Retângulo)")
print(f"Pixels Reais (Esperado): {area_real_sintetica}")
print(f"Pixels Detectados: {px_sint}")
print(f"Erro Percentual do Método: {erro_percentual:.2f}%")
cv2.imwrite("RESULTADO_sintetica.jpg", res_sint)
print("-> Arquivo gerado com sucesso: 'RESULTADO_sintetica.jpg'\n")