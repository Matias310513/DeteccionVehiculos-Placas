import os
import cv2
import datetime
from ultralytics import YOLO

# ==========================================
# CARGA DE MODELOS DE INTELIGENCIA ARTIFICIAL
# ==========================================
# modelo YOLO para deteccion y seguimiento de vehiculos
modelo_vehiculos = YOLO('yolov8n.pt')

# modelo YOLO para localizacion de placas
modelo_placas = YOLO('licence_plate_detector.pt')

# carpetas para guardar los recortes
carpeta_vehiculos = 'vehiculos'
carpeta_placas = 'placas'

os.makedirs(carpeta_vehiculos, exist_ok=True)
os.makedirs(carpeta_placas, exist_ok=True)

# ==========================================
# CONFIGURACION DE FILTROS DE TAMANO Y NITIDEZ
# ==========================================
# filtros minimos para vehiculos
ANCHO_MIN_VEHICULO = 80
ALTO_MIN_VEHICULO = 80
NITIDEZ_MIN_VEHICULO = 50.0

# filtros minimos para matriculas
ANCHO_MIN_PLACA = 35
ALTO_MIN_PLACA = 12
NITIDEZ_MIN_PLACA = 40.0
CONF_MIN_PLACA = 0.5  # confianza minima para descartar falsas detecciones

# contadores de resultados guardados
count_vehiculos = 0
count_placas = 0

# ==========================================
# FUNCIONES PARA NITIDEZ Y PUNTUACION
# ==========================================
def calcular_nitidez(imagen):
    # calcula la nitidez utilizando la varianza del operador laplaciano en escala de grises.
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gris, cv2.CV_64F).var()

def calcular_puntuacion(ancho, alto, nitidez):
    # calcula la puntuacion combinando el area del recorte y su nivel de nitidez.
    area = ancho * alto
    return area * nitidez

# diccionarios para conservar unicamente la captura con mayor puntuacion por ID
mejores_vehiculos = {}
mejores_placas = {}

# ==========================================
# APERTURA Y PROCESAMIENTO DEL VIDEO
# ==========================================
ruta_video = 'video.mp4'
cap = cv2.VideoCapture(ruta_video)

if not cap.isOpened():
    print('Error: No se pudo abrir el archivo de video.')
    exit()

print('Iniciando procesamiento de video...')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # copia limpia para extraer recortes puros sin rectangulos ni texto visual
    frame_limpio = frame.copy()

    # --------------------------------------
    # DETECCION Y SEGUIMIENTO DE VEHICULOS
    # --------------------------------------
    resultados_veh = modelo_vehiculos.track(frame, persist=True, verbose=False)
    vehiculos_detectados = []

    if resultados_veh[0].boxes is not None and resultados_veh[0].boxes.id is not None:
        boxes = resultados_veh[0].boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())

            # clase 2 corresponde a automoviles (car) en el dataset COCO
            if cls_id == 2:
                track_id = int(boxes.id[i].item())
                caja = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3])

                vehiculos_detectados.append((track_id, x1, y1, x2, y2))

                w = x2 - x1
                h = y2 - y1

                # filtrar por dimension minima
                if w >= ANCHO_MIN_VEHICULO and h >= ALTO_MIN_VEHICULO:
                    recorte = frame_limpio[y1:y2, x1:x2]

                    if recorte.size > 0:
                        nitidez = calcular_nitidez(recorte)

                        if nitidez >= NITIDEZ_MIN_VEHICULO:
                            score = calcular_puntuacion(w, h, nitidez)

                            # conservar solo la captura con mayor puntuacion para el vehiculo
                            if (track_id not in mejores_vehiculos) or (score > mejores_vehiculos[track_id]['score']):
                                mejores_vehiculos[track_id] = {
                                    'score': score,
                                    'img': recorte,
                                    'w': w,
                                    'h': h,
                                    'nitidez': nitidez
                                }

                # anotaciones en vivo en pantalla
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --------------------------------------
    # DETECCION Y ASOCIACION DE MATRICULAS
    # --------------------------------------
    resultados_placas = modelo_placas(frame, conf=CONF_MIN_PLACA, verbose=False)

    if resultados_placas[0].boxes is not None:
        boxes_p = resultados_placas[0].boxes

        for i in range(len(boxes_p)):
            caja_p = boxes_p.xyxy[i].cpu().numpy()
            px1, py1, px2, py2 = int(caja_p[0]), int(caja_p[1]), int(caja_p[2]), int(caja_p[3])

            pw = px2 - px1
            ph = py2 - py1

            # filtrar por dimensiones minimas
            if pw >= ANCHO_MIN_PLACA and ph >= ALTO_MIN_PLACA:
                relacion_aspecto = pw / float(ph)

                # validar relacion de aspecto rectangular de la placa real (mas ancha que alta)
                if 1.5 <= relacion_aspecto <= 5.5:
                    centro_x = (px1 + px2) // 2
                    centro_y = (py1 + py2) // 2

                    # comprobar si el centro de la placa esta dentro de algun vehiculo
                    for v_id, vx1, vy1, vx2, vy2 in vehiculos_detectados:
                        if vx1 <= centro_x <= vx2 and vy1 <= centro_y <= vy2:
                            recorte_placa = frame_limpio[py1:py2, px1:px2]

                            if recorte_placa.size > 0:
                                p_nitidez = calcular_nitidez(recorte_placa)

                                if p_nitidez >= NITIDEZ_MIN_PLACA:
                                    p_score = calcular_puntuacion(pw, ph, p_nitidez)

                                    # conservar la mejor captura de placa asociada al ID del vehiculo
                                    if (v_id not in mejores_placas) or (p_score > mejores_placas[v_id]['score']):
                                        mejores_placas[v_id] = {
                                            'score': p_score,
                                            'img': recorte_placa,
                                            'w': pw,
                                            'h': ph,
                                            'nitidez': p_nitidez
                                        }

                    # dibujar recuadro de la placa en visualizacion
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

    # reproduccion en pantalla con detecciones (finaliza con tecla Esc)
    cv2.imshow('Sistema Inteligente de Deteccion', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# SELECCION Y ALMACENAMIENTO DE CAPTURAS
# ==========================================
marca_tiempo = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print('\n--- Guardando capturas de vehiculos ---')
for v_id, datos in mejores_vehiculos.items():
    count_vehiculos += 1
    nombre_archivo = f'vehiculo_{v_id}_{marca_tiempo}.png'
    ruta_guardado = os.path.join(carpeta_vehiculos, nombre_archivo)
    cv2.imwrite(ruta_guardado, datos['img'])
    print(f'Guardado: {nombre_archivo} | Ancho: {datos["w"]}px, Alto: {datos["h"]}px | Nitidez: {datos["nitidez"]:.2f}')

print('\n--- Guardando capturas de matriculas ---')
for v_id, datos in mejores_placas.items():
    count_placas += 1
    nombre_archivo = f'placa_vehiculo_{v_id}_{marca_tiempo}.png'
    ruta_guardado = os.path.join(carpeta_placas, nombre_archivo)

    img_placa = datos['img']
    # redimensionar suavemente si la placa es muy pequena para legibilidad visual
    if datos['w'] < 120:
        factor = 120.0 / datos['w']
        nuevo_w = 120
        nuevo_h = int(datos['h'] * factor)
        img_placa = cv2.resize(img_placa, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(ruta_guardado, img_placa)
    print(f'Guardado: {nombre_archivo} | Ancho: {datos["w"]}px, Alto: {datos["h"]}px | Nitidez: {datos["nitidez"]:.2f}')

# ==========================================
# PRESENTACION DEL RESUMEN FINAL
# ==========================================
print('\n================ RESUMEN FINAL ================')
print(f'Total de vehiculos guardados en \'{carpeta_vehiculos}/\': {count_vehiculos}')
print(f'Total de matriculas guardadas en \'{carpeta_placas}/\': {count_placas}')
print('===============================================')