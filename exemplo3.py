import cv2

"""
Moisés 11/05/202
Faz a detecção de imagens usando a webcam
"""

# Faz a referencia para webcam
video = cv2.VideoCapture(0)

# Carrega o classificador
classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

while True:
    # Faz a leitura da webcan
    conectado, frame = video.read()

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(frame_cinza, minSize=(60, 60))

    # Itera sobre a matriz para desenhar o retangulo em cada face encontrada
    for (ponto_x, ponto_y, largura, altura) in faces_detectadas:
        largunra_borda = 2
        cor_rgb = (0, 0, 255)
        medidas_retangulo = (ponto_x + largura, ponto_y + altura)
        ponto_x_y = (ponto_x, ponto_y)
        cv2.rectangle(frame, ponto_x_y, medidas_retangulo, cor_rgb, largunra_borda)

    # Carrega para janela o video
    cv2.imshow('Video', frame)

    # Usa a tecla q para fechar a janela com o video
    if cv2.waitKey(1) == ord('q'):
        break

# Libera a memória e destroi as janelas
video.release()
cv2.destroyAllWindows()
