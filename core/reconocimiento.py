from config import TOLERANCIA

def extraer_dos_manos(multi_hand_landmarks):

    datos = []

    for handLms in multi_hand_landmarks:

        wrist = handLms.landmark[0]

        for lm in handLms.landmark:
            x = lm.x - wrist.x
            y = lm.y - wrist.y
            z = lm.z - wrist.z

            datos.extend([x, y, z])

    return datos

def extraer_landmarks(handLms):

    datos = []

    wrist = handLms.landmark[0]

    for lm in handLms.landmark:

        x = lm.x - wrist.x
        y = lm.y - wrist.y
        z = lm.z - wrist.z

        datos.extend([x, y, z])

    return datos

def comparar_letra(datos_actuales, dataset):

    mejor_letra = None
    mejor_distancia = float("inf")

    for letra, ejemplos in dataset.items():

        for ejemplo in ejemplos:

            if len(ejemplo) != len(datos_actuales):
                continue

            distancia_total = 0

            for a, b in zip(datos_actuales, ejemplo):
                distancia_total += abs(a - b)

            distancia_promedio = distancia_total / len(datos_actuales)

            if distancia_promedio < mejor_distancia:
                mejor_distancia = distancia_promedio
                mejor_letra = letra

    if mejor_distancia < TOLERANCIA:
        return mejor_letra

    return None

def coincide_objetivo( datos_actuales, letra_objetivo, dataset):

    if letra_objetivo not in dataset:
        return False

    ejemplos = dataset[letra_objetivo]

    mejor_distancia = float("inf")

    for ejemplo in ejemplos:

        if len(ejemplo) != len(datos_actuales):
            continue

        distancia_total = 0

        for a, b in zip(datos_actuales, ejemplo):
            distancia_total += abs(a - b)

        distancia_promedio = (
            distancia_total /
            len(datos_actuales)
        )

        mejor_distancia = min(
            mejor_distancia,
            distancia_promedio
        )

    return mejor_distancia < TOLERANCIA