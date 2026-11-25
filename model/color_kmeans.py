"""
color_kmeans.py
---------------
Obtiene el color dominante agrupando píxeles con k-means implementado desde cero.
"""

import colorsys
import numpy as np
from PIL import Image

NEUTROS = {"negro", "blanco", "gris", "beige", "café", "azul marino"}


def kmeans(datos, k=4, max_iter=50, tol=1e-3, semilla=0):
    """
    Agrupa datos (n x 3, valores RGB) en k grupos usando k-means.
    
    Returns:
        (centroides k x 3, etiquetas n)
    """
    rng = np.random.default_rng(semilla)
    datos = datos.astype(np.float64)
    n = len(datos)

    # Inicialización k-means++
    centroides = [datos[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((datos[:, None, :] - np.array(centroides)[None]) ** 2).sum(-1), axis=1)
        prob = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centroides.append(datos[rng.choice(n, p=prob)])
    centroides = np.array(centroides)

    for _ in range(max_iter):
        # Asignación
        distancias = ((datos[:, None, :] - centroides[None]) ** 2).sum(-1)
        etiquetas = distancias.argmin(axis=1)

        # Actualización
        nuevos = np.array([
            datos[etiquetas == j].mean(axis=0) if np.any(etiquetas == j) else centroides[j]
            for j in range(k)
        ])

        # Convergencia
        movimiento = np.abs(nuevos - centroides).max()
        centroides = nuevos
        if movimiento < tol:
            break

    return centroides, etiquetas


def cargar_pixeles(imagen, tamano=120, recorte=0.15):
    """
    Reduce la imagen y quita margen del borde (donde suele estar el fondo).
    
    Returns:
        (pixeles_centro n x 3, pixeles_borde m x 3)
    """
    if not isinstance(imagen, Image.Image):
        imagen = Image.open(imagen)
    imagen = imagen.convert("RGB")
    imagen.thumbnail((tamano, tamano))
    arr = np.asarray(imagen)
    h, w, _ = arr.shape
    dy, dx = int(h * recorte), int(w * recorte)

    centro = arr[dy:h - dy, dx:w - dx].reshape(-1, 3)
    borde = np.concatenate([
        arr[:dy].reshape(-1, 3), arr[h - dy:].reshape(-1, 3),
        arr[:, :dx].reshape(-1, 3), arr[:, w - dx:].reshape(-1, 3),
    ])
    return centro, borde


def color_dominante(imagen, k=4):
    """
    Devuelve el color dominante (R, G, B) y la paleta completa.
    Ignora el fondo comparando con los bordes de la imagen.
    """
    centro, borde = cargar_pixeles(imagen)
    centroides, etiquetas = kmeans(centro, k=k)
    tamanos = np.bincount(etiquetas, minlength=k)

    color_fondo = np.median(borde, axis=0)
    dist_fondo = np.linalg.norm(centroides - color_fondo, axis=1)

    orden = np.argsort(-tamanos)
    paleta = [(tuple(int(c) for c in centroides[j]), tamanos[j] / tamanos.sum()) for j in orden]

    for j in orden:
        if dist_fondo[j] > 40:
            return tuple(int(c) for c in centroides[j]), paleta
    
    # Si todo se parece al fondo
    j = orden[0]
    return tuple(int(c) for c in centroides[j]), paleta


NOMBRES_TONO = [
    (15, "rojo"), (40, "naranja"), (65, "amarillo"), (160, "verde"),
    (195, "turquesa"), (255, "azul"), (290, "morado"), (335, "rosa"), (360, "rojo"),
]


def describir_color(rgb):
    """
    Convierte RGB a HSV y clasifica el color.
    
    Returns:
        dict con nombre, tono, saturacion, brillo, neutro, fuerte
    """
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    tono = h * 360

    nombre = None
    if v < 0.22:
        nombre = "negro"
    elif s < 0.15 and v > 0.85:
        nombre = "blanco"
    elif s < 0.15:
        nombre = "gris"
    elif 15 <= tono <= 50 and s < 0.45 and v > 0.6:
        nombre = "beige"
    elif 5 <= tono <= 50 and v < 0.65:
        nombre = "café"
    elif 195 <= tono <= 260 and v < 0.5:
        nombre = "azul marino"

    if nombre is None:
        nombre = next(n for limite, n in NOMBRES_TONO if tono <= limite)
        if nombre in ("rojo", "rosa"):
            if v > 0.55 and s < 0.6:
                nombre = "rosa"
            elif v < 0.5:
                nombre = "vino"
        elif nombre == "azul" and v > 0.75 and s < 0.5:
            nombre = "celeste"

    neutro = nombre in NEUTROS
    fuerte = (not neutro) and s > 0.6 and v > 0.5
    
    return {
        "nombre": nombre, 
        "tono": tono, 
        "saturacion": s, 
        "brillo": v,
        "neutro": neutro, 
        "fuerte": fuerte,
    }


def rgb_a_hex(rgb):
    """Convierte RGB a hexadecimal."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)