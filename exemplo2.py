import cv2

"""
Moisés 08/05/202
Faz a detecção dos olhos nas imagens
"""

# Carrega o classificador para as faces
classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Carrega o classificador para os olhos
classificador_olhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
# Carrega a imagem
imagem = cv2.imread('familia/foto1.jpg')

# Converte para cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Variavel que armzena a matriz de dados encontrados
faces_e_olhos_detectados = classificador.detectMultiScale(imagem_cinza)

# Itera sobre a matriz para desenhar o retangulo em cada face encontrada
for (ponto_x, ponto_y, largura, altura) in faces_e_olhos_detectados:

    largunra_borda = 3
    cor_rgb = (0, 0, 255)
    medidas_retangulo = (ponto_x + largura, ponto_y + altura)
    ponto_x_y = (ponto_x, ponto_y)
    cv2.rectangle(imagem, ponto_x_y, medidas_retangulo, cor_rgb, largunra_borda)

    # Cria a referencia para a face que foi detectada para encontrar os olhos
    regiao_face_detectada = imagem[ponto_y:ponto_y + altura, ponto_x:ponto_x + largura]
    regiao_cinza_olho = cv2.cvtColor(regiao_face_detectada, cv2.COLOR_BGR2GRAY)
    olhos_detectados = classificador_olhos.detectMultiScale(regiao_cinza_olho, scaleFactor=1.25, minNeighbors=1)
    print("OLHOS DETECTADOS", olhos_detectados)

    # Itera sobre os resultados para desenhar o retângulo dos olhos
    for (olhos_x, olhos_y, olhos_largura, olhos_altura) in olhos_detectados:

        largura_borda_olho = 2
        cor_rgb_olho = (0, 0, 0)
        medidas_retangulo_olho = (olhos_x + olhos_largura, olhos_y + olhos_altura)
        ponto_x_y_olho = (olhos_x, olhos_y)

        cv2.rectangle(regiao_face_detectada, ponto_x_y_olho, medidas_retangulo_olho, cor_rgb_olho, largura_borda_olho)

cv2.imshow("Faces e Olhos Encontrados", imagem)
cv2.waitKey()
