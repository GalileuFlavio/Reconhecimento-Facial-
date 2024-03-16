# Reconhecimento-Facial-
Script : Reconhecimento Facial com OpenCV e Deep Learning

import cv2

# Carregar o modelo de detecção facial
modelo_deteccao = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Carregar a imagem
imagem = cv2.imread("imagem.jpg")

# Pré-processamento da imagem
altura, largura = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Passar o blob pela rede neural
modelo_deteccao.setInput(blob)
deteccoes = modelo_deteccao.forward()

# Loop pelas detecções e desenhar caixas ao redor dos rostos
for i in range(deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]
    if confianca > 0.5:
        caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
        (x, y, x2, y2) = caixa.astype("int")
        cv2.rectangle(imagem, (x, y), (x2, y2), (0, 255, 0), 2)

# Exibir a imagem com as detecções
cv2.imshow("Detecção Facial", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
