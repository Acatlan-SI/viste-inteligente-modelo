"""
caracteristicas.py
------------------
Extracción de features combinando CNN (MobileNetV2) y características manuales.
"""

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from model.color_kmeans import color_dominante, describir_color

# Configuración CNN
MODELO_ONNX_PATH = "models/mobilenetv2-12.onnx"
_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Sesión ONNX global para no recargarla en cada request
_sesion_cnn = None

def _obtener_sesion_cnn():
    global _sesion_cnn
    if _sesion_cnn is None:
        try:
            _sesion_cnn = ort.InferenceSession(MODELO_ONNX_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo ONNX: {e}")
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
    return x.transpose(2, 0, 1)[None] # Formato (1, 3, 224, 224)

def obtener_vector_cnn(imagen_input):
    """Devuelve el vector de 1000 características de la CNN."""
    img = abrir(imagen_input) # Usamos la función 'abrir' ya definida
    sesion = _obtener_sesion_cnn()
    nombre_entrada = sesion.get_inputs()[0].name
    salida = sesion.run(None, {nombre_entrada: _preparar_imagen_cnn(img)})[0]
    return np.asarray(salida, dtype=np.float32).reshape(-1)[:1000]

# ... (Mantener las funciones _histograma_hsv, _forma y extraer igual que antes) ...

def extraer_features_array(imagen_input):
    """
    Extrae el vector completo de 1067 características (CNN + Manual).
    """
    # 1. Features Manuales (67)
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
    
    # 2. Features CNN (1000)
    cnn_array = obtener_vector_cnn(imagen_input)
    
    # 3. Concatenar (1067 total)
    return np.hstack([cnn_array, manual_array]).reshape(1, -1)