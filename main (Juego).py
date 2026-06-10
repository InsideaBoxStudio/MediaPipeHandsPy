import cv2
import mediapipe as mp
import time
import random
import pygame

from config import FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAMES_NECESARIOS
from core.camara import crear_camara
from core.dataset import cargar_dataset
from core.reconocimiento import (
    extraer_landmarks,
    extraer_dos_manos,
    coincide_objetivo
)

# =========================
# VARIABLES
# =========================

isRuning = True
frames_correctos = 0
letra_objetivo = None
puntaje = 0

dataset = cargar_dataset()

video = crear_camara()

FRAME_TIME = 1 / FPS

# =========================
# MEDIAPIPE
# =========================

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False, # False para videp, True para imagenes
    max_num_hands=2, # numero de manos a detectar
    min_detection_confidence=0.7, # confianza minima para detectar una mano
    min_tracking_confidence=0.7 # confianza minima para seguir una mano detectada
)

mpDraw = mp.solutions.drawing_utils # variable para dibujar los landmarks

# =========================
# SONIDOS
# =========================

pygame.mixer.init()
sonido_acierto = pygame.mixer.Sound("assets\sound effects\Acierto.mp3")

# =========================
# LOOP
# =========================

while True:
    inicio = time.time() # tiempo de inicio del loop

    # -------------------------
    # CAPTURA
    # -------------------------

    for _ in range(2):
        video.grab()

    ret, frame = video.read() # leer un frame de la camara

    if not ret: # si no se pudo leer el frame
        break # salir del loop

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) # redimencionar el frame
    frame = cv2.flip(frame, 1) # dar vuelta el frame para que sea espejo
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convertir el frame a RGB para mediapipe

    # -------------------------
    # MEDIAPIPE
    # -------------------------

    resultado = hands.process(imgRGB) # procesar el frame con mediapipe para detectar las manos
    
    if resultado.multi_hand_landmarks: # si se detectaron manos

        for handLms in resultado.multi_hand_landmarks: # para cada mano detectada

            mpDraw.draw_landmarks( # dibujar los landmarks
                frame,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    # -------------------------
    # Juego
    # -------------------------

    if letra_objetivo is None: # si no hay una letra objetivo
        letra_objetivo = random.choice( list(dataset.keys()) )

    datos_actuales = None

    if resultado.multi_hand_landmarks: # si se detectaron manos

        cantidad_manos = len( resultado.multi_hand_landmarks ) # guardar cuantas manos se detectaron

        if cantidad_manos == 2: # si se detectaron 2 manos
            datos_actuales = extraer_dos_manos( resultado.multi_hand_landmarks ) # extraer los datos de ambas manos
        else: # si se detecto solo una mano
            datos_actuales = extraer_landmarks(  resultado.multi_hand_landmarks[0] ) # extraer los datos de la mano

        if datos_actuales is None:
            frames_correctos = 0

        elif coincide_objetivo( datos_actuales, letra_objetivo, dataset ): # si el jugador hizo la seña correcta
            frames_correctos += 1
    
    if frames_correctos >= FRAMES_NECESARIOS: # si el jugador mantuvo la seña correcta por suficientes frames

        puntaje += 1
        sonido_acierto.play()

        letra_objetivo = None

        frames_correctos = 0

    # -------------------------
    # TECLADO
    # -------------------------

    tecla = cv2.waitKey(1) & 0xFF # esperar por una tecla

    if tecla == ord('.') or tecla == 27: # si se preciona '.' o 'ESC'
        break # cerrar el juego

    # -------------------------
    # UI
    # -------------------------

    if letra_objetivo is not None:
        cv2.putText(
            frame,
            f"Letra: {letra_objetivo}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
    
    cv2.putText(
        frame,
        f"Puntaje: {puntaje}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    if isRuning:
        cv2.imshow("Reconocimiento", frame)

    if cv2.getWindowProperty(
        "Reconocimiento",
        cv2.WND_PROP_AUTOSIZE
    ) < 1: # si la ventana se cerro manualmente
        isRuning = False
        break # cerrar el juego

    # -------------------------
    # FPS
    # -------------------------

    tiempo = time.time() - inicio

    if tiempo < FRAME_TIME:
        time.sleep(FRAME_TIME - tiempo)


# =========================
# LIMPIEZA
# =========================

video.release()
cv2.destroyAllWindows()