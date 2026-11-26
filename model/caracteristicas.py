"""
caracteristicas.py
------------------
Extracción de features combinando CNN (MobileNetV2) y características manuales.
"""

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from model.color_kmeans import color_dominante, describir_color

MODELO_ONNX_PATH = "models/mobilenetv2-12.onnx"  
_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sesion_cnn = None

def _obtener_sesion_cnn():
    """Carga y reutiliza la sesión ONNX de MobileNetV2."""
    global _sesion_cnn
    if _sesion_cnn is None:
        try:
            _sesion_cnn = ort.InferenceSession(MODELO_ONNX_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo ONNX en {MODELO_ONNX_PATH}: {e}")
    return _sesion_cnn

def _preparar_imagen_cnn(img):
    """Prepara la imagen para MobileNetV2 (resize 224, normalización ImageNet)."""
    ancho, alto = img.size
    escala = 256 / min(ancho, alto)
    img_resized = img.resize((max(224, round(ancho * escala)), max(224, round(alto * escala))), Image.BILINEAR)
    ancho, alto = img_resized.size
    izq, arr = (ancho - 224) // 2, (alto - 224) // 2
    img_crop = img_resized.crop((izq, arr, izq + 224, arr + 224))
    
    x = (np.asarray(img_crop, dtype=np.float32) / 255.0 - _MEDIA) / _DESV
    return x.transpose(2, 0, 1)[None]

def obtener_vector_cnn(imagen_input):
    """Devuelve el vector de 1000 características de la CNN."""
    img = abrir(imagen_input)
    sesion = _obtener_sesion_cnn()
    nombre_entrada = sesion.get_inputs()[0].name
    salida = sesion.run(None, {nombre_entrada: _preparar_imagen_cnn(img)})[0]
    return np.asarray(salida, dtype=np.float32).reshape(-1)[:1000]


def abrir(imagen_input):
    """Abre una imagen desde bytes, ruta o objeto PIL."""
    if isinstance(imagen_input, Image.Image):
        img = imagen_input
    elif isinstance(imagen_input, bytes):
        from io import BytesIO
        img = Image.open(BytesIO(imagen_input))
    else:
        img = Image.open(imagen_input)
    
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _histograma_hsv(img):
    """Histogramas normalizados de tono, saturación y brillo del centro de la foto."""
    BINS_H, BINS_S, BINS_V = 12, 4, 4
    w, h = img.size
    centro = img.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
    hsv = np.asarray(centro.convert("HSV")).reshape(-1, 3).astype(float)
    hh = np.histogram(hsv[:, 0], bins=BINS_H, range=(0, 256))[0]
    ss = np.histogram(hsv[:, 1], bins=BINS_S, range=(0, 256))[0]
    vv = np.histogram(hsv[:, 2], bins=BINS_V, range=(0, 256))[0]
    n = len(hsv)
    return np.concatenate([hh, ss, vv]) / n


def _forma(img):
    """Descriptor de forma simplificado (tipo HOG)."""
    CELDAS, BINS_ORIENT = 2, 9
    gris = np.asarray(img.convert("L").resize((64, 64)), dtype=float) / 255
    gy, gx = np.gradient(gris)
    magnitud = np.hypot(gx, gy)
    angulo = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    tam = 64 // CELDAS
    desc = []
    for i in range(CELDAS):
        for j in range(CELDAS):
            m = magnitud[i * tam:(i + 1) * tam, j * tam:(j + 1) * tam].ravel()
            a = angulo[i * tam:(i + 1) * tam, j * tam:(j + 1) * tam].ravel()
            hist = np.histogram(a, bins=BINS_ORIENT, range=(0, 180), weights=m)[0]
            desc.append(hist / (np.linalg.norm(hist) + 1e-6))
    densidad = float((magnitud > 0.1).mean())
    return np.concatenate(desc), densidad


def extraer(imagen_input):
    """
    Extrae todas las características manuales de una imagen.
    Returns: dict con características + metadatos de color
    """
    img = abrir(imagen_input)
    proporcion = img.height / img.width
    img.thumbnail((256, 256))

    rgb, _ = color_dominante(img, k=4)
    c = describir_color(rgb)
    rad = np.radians(c["tono"])

    BINS_H, BINS_S, BINS_V = 12, 4, 4
    CELDAS, BINS_ORIENT = 2, 9
    COLUMNAS_MODELO = (
        ["dom_r", "dom_g", "dom_b", "dom_tono_sin", "dom_tono_cos", "dom_saturacion",
         "dom_brillo", "es_neutro", "es_fuerte"]
        + [f"hist_h{i}" for i in range(BINS_H)]
        + [f"hist_s{i}" for i in range(BINS_S)]
        + [f"hist_v{i}" for i in range(BINS_V)]
        + [f"forma_{i}" for i in range(CELDAS * CELDAS * BINS_ORIENT)]
        + ["proporcion", "densidad_bordes"]
    )

    fila = {
        "dom_r": rgb[0], "dom_g": rgb[1], "dom_b": rgb[2],
        "dom_tono_sin": float(np.sin(rad)), "dom_tono_cos": float(np.cos(rad)),
        "dom_saturacion": c["saturacion"], "dom_brillo": c["brillo"],
        "es_neutro": int(c["neutro"]), "es_fuerte": int(c["fuerte"]),
    }
    
    for nombre, valor in zip(COLUMNAS_MODELO[9:9 + 20], _histograma_hsv(img)):
        fila[nombre] = float(valor)
    
    forma, densidad = _forma(img)
    for i, valor in enumerate(forma):
        fila[f"forma_{i}"] = float(valor)
    
    fila["proporcion"] = proporcion
    fila["densidad_bordes"] = densidad
    fila["color_nombre"] = c["nombre"]
    fila["dom_tono"] = c["tono"]
    
    return fila


def extraer_features_array(imagen_input, input_type="CNN (1000)"):
    """
    Extrae el vector de características según el tipo de entrada del modelo.
    
    Args:
        imagen_input: bytes, str o PIL.Image
        input_type: "Manual (67)", "CNN (1000)" o "CNN + manual (1067)"
        
    Returns:
        numpy array con las features en el formato correcto
    """
    features_dict = extraer(imagen_input)
    
    BINS_H, BINS_S, BINS_V = 12, 4, 4
    CELDAS, BINS_ORIENT = 2, 9
    COLUMNAS_MODELO = (
        ["dom_r", "dom_g", "dom_b", "dom_tono_sin", "dom_tono_cos", "dom_saturacion",
         "dom_brillo", "es_neutro", "es_fuerte"]
        + [f"hist_h{i}" for i in range(BINS_H)]
        + [f"hist_s{i}" for i in range(BINS_S)]
        + [f"hist_v{i}" for i in range(BINS_V)]
        + [f"forma_{i}" for i in range(CELDAS * CELDAS * BINS_ORIENT)]
        + ["proporcion", "densidad_bordes"]
    )
    
    manual_list = [features_dict[col] for col in COLUMNAS_MODELO]
    manual_array = np.array(manual_list, dtype=np.float32)
    
    cnn_array = obtener_vector_cnn(imagen_input)
    
    if input_type == "Manual (67)":
        return manual_array.reshape(1, -1)
    elif input_type == "CNN (1000)":
        return cnn_array.reshape(1, -1)
    elif input_type == "CNN + manual (1067)":
        return np.hstack([cnn_array, manual_array]).reshape(1, -1)
    else:
        return cnn_array.reshape(1, -1)


def get_onehot_tipo(tipo_idx, num_clases=5):
    """Genera one-hot encoding para el tipo de prenda."""
    onehot = np.zeros((1, num_clases))
    onehot[0, int(tipo_idx)] = 1
    return onehot