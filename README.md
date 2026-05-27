# MediaPipeHandsPy
MediaPipe en Python

#necesario (installalo con "pip install 'copia y pega los siguientes nombres de complementos'"):

mediapipe==0.10.21
opencv-python
pygame
pyinstaller

#compilar en un ejecutable:

python -m PyInstaller --onedir --collect-all mediapipe --collect-all pygame --add-data "musical notes;musical notes" finalcount.py