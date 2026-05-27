import os
import sys

import cv2
import mediapipe as mp
import pygame

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

pygame.mixer.init()

sounds = [
    pygame.mixer.Sound(resource_path(r"musical notes\#do.WAV")),
    pygame.mixer.Sound(resource_path(r"musical notes\#fa.WAV")),
    pygame.mixer.Sound(resource_path(r"musical notes\#sol.WAV")),
    pygame.mixer.Sound(resource_path(r"musical notes\la.WAV")),
    pygame.mixer.Sound(resource_path(r"musical notes\#re.WAV")),
    pygame.mixer.Sound(resource_path(r"musical notes\si.WAV")),
]

def is_finger_down(landmarks, finger_tip, finger_msp):
    return landmarks[finger_tip].y > landmarks[finger_msp].y

capture = cv2.VideoCapture(0) #variable con la camara (se puede pasar un video poniendo la ruta)

with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    max_num_hands = 2) as hands:

    finger_state = [False]*6

    while capture.isOpened(): #obtener todos los frames de la camara/video
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame,1) #dar vuelta el frame
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #cambiar de BlueGreenRed a RedGreenBlue
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks: #si detecta almenos una mano
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #dibujar los puntos y conecciones

                finger_tips = [8, 12, 16] #lista de punta de los dedos
                finger_mcp = [5, 9, 13] #lista de nudillos

                for i in range(3):
                    finger_index = i + h * 3
                    if is_finger_down(hand_landmarks.landmark, finger_tips[i], finger_mcp[i]):
                        if not finger_state[finger_index]:
                            sounds[finger_index].play()
                            finger_state[finger_index] = True
                    else:
                        finger_state[finger_index] = False


        cv2.imshow('HandDetection', frame) #mostrar en pantalla/ventana

        if cv2.waitKey(1) & 0xFF == 27: #si el usuario presiona Esq: terminar proceso
            break

capture.release() #detener captura
cv2.destroyAllWindows() #destruir ventana