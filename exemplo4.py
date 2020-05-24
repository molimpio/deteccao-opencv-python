import cv2

"""
Moisés 11/05/202
Faz a detecção de imagens de gatos
"""

# Carrega o classificador
classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')

# Carrega a imagem
imagem = cv2.imread('outros/gato2.jpg')

# Converte para cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Variavel que armzena a matriz de dados encontrados
# Faz a detecção, usa scaleFactor (fator de escala das imagens para detecção)
# Usando quando não consegue realizar a detecção por padrão sem scaleFactor
# minNeighbors qtde de vizinhos cada retângulo deve ter para mantê-lo, quando as imagens estão lado a lado
# minSize é o tamanho mínimo da face para detecção
faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.03)

print("QTDE FACES DETECTADAS", len(faces_detectadas))
print("MATRIZ DE DADOS", faces_detectadas)

# Itera sobre a matriz para desenhar o retangulo em cada face encontrada
for (ponto_x, ponto_y, largura, altura) in faces_detectadas:

    largunra_borda = 2
    cor_rgb = (0, 0, 255)
    medidas_retangulo = (ponto_x + largura, ponto_y + altura)
    ponto_x_y = (ponto_x, ponto_y)
    cv2.rectangle(imagem, ponto_x_y, medidas_retangulo, cor_rgb, largunra_borda)

cv2.imshow("Faces Encontradas", imagem)
cv2.waitKey()
