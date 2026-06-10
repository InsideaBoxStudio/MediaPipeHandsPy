import cv2
import mediapipe as mp
import time

from config import FPS
from core.camara import crear_camara
from core.dataset import cargar_dataset, guardar_letra
from core.reconocimiento import (
    extraer_landmarks,
    comparar_letra,
    extraer_dos_manos
)

# =========================
# VARIABLES
# =========================

pausar = False
texto_actual = ""
frame_pausado = None

dataset = cargar_dataset()

video = crear_camara()

FRAME_TIME = 1 / FPS

# =========================
# MEDIAPIPE
# =========================

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# =========================
# LOOP
# =========================

while True:

    inicio = time.time()

    # -------------------------
    # CAPTURA
    # -------------------------

    if not pausar:

        for _ in range(2):
            video.grab()

        ret, frame = video.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        frame_pausado = frame.copy()

    else:

        if frame_pausado is None:
            continue

        frame = frame_pausado.copy()

    # -------------------------
    # MEDIAPIPE
    # -------------------------

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado = hands.process(imgRGB)

    letra_detectada = None
    datos_actuales = None

    if resultado.multi_hand_landmarks:

        for handLms in resultado.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                frame,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

        cantidad_manos = len(resultado.multi_hand_landmarks)

        if cantidad_manos == 2 and resultado.multi_handedness:

            pares = list(zip(
                resultado.multi_hand_landmarks,
                resultado.multi_handedness
            ))

            pares.sort(
                key=lambda p:
                0 if p[1].classification[0].label == "Left"
                else 1
            )

            manos_ordenadas = [p[0] for p in pares]

            datos_actuales = extraer_dos_manos(
                manos_ordenadas
            )

        else:

            datos_actuales = extraer_landmarks(
                resultado.multi_hand_landmarks[0]
            )

        letra_detectada = comparar_letra(
            datos_actuales,
            dataset
        )

    # -------------------------
    # TECLADO
    # -------------------------

    tecla = cv2.waitKey(1) & 0xFF

    # Salir
    if tecla == ord('.') or tecla == 27:
        break

    # Pausar / Reanudar
    if tecla == ord(' '):

        if not pausar:
            frame_pausado = frame.copy()

        pausar = not pausar

    # Escribir palabra mientras está pausado
    if pausar:

        if ord('a') <= tecla <= ord('z'):
            texto_actual += chr(tecla).upper()

        elif ord('A') <= tecla <= ord('Z'):
            texto_actual += chr(tecla)

        elif ord('0') <= tecla <= ord('9'):
            texto_actual += chr(tecla)

        elif tecla == 8:  # Backspace
            texto_actual = texto_actual[:-1]

        elif tecla in (10, 13):  # Enter

            if texto_actual and datos_actuales is not None:

                guardar_letra(
                    dataset,
                    texto_actual,
                    datos_actuales
                )

                print(f"Guardado: {texto_actual}")

            texto_actual = ""
            pausar = False

    # Guardar letras individuales cuando NO está pausado
    else:

        if (
            ord('a') <= tecla <= ord('z')
            and datos_actuales is not None
        ):

            letra = chr(tecla).upper()

            guardar_letra(
                dataset,
                letra,
                datos_actuales
            )

    # -------------------------
    # UI
    # -------------------------

    if letra_detectada:

        cv2.putText(
            frame,
            f"Detectado: {letra_detectada}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    estado = "PAUSADO" if pausar else "ACTIVO"

    cv2.putText(
        frame,
        f"Estado: {estado}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Etiqueta: {texto_actual}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Cerrar ventana manualmente
    cv2.imshow("Reconocimiento", frame)

    try:
        if cv2.getWindowProperty(
            "Reconocimiento",
            cv2.WND_PROP_AUTOSIZE
        ) < 0:
            break
            print("Saliendo del loop...")
    except:
        print("Excepción al consultar ventana")
        break

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