import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX

def crear_camara():

    video = cv2.VideoCapture(CAMERA_INDEX)

    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return video