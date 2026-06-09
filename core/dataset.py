import json
import os
from config import JSON_FILE

def cargar_dataset():

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return json.load(f)

    return {}

def guardar_letra(dataset, letra, datos):

    if letra not in dataset:
        dataset[letra] = []

    dataset[letra].append(datos)

    with open(JSON_FILE, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Letra {letra} guardada")