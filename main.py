"""
main.py
-------
Servicio backend de ML para "Viste Inteligente".

Endpoints:
  GET  /health      - Estado del servicio
  POST /classify    - Clasificación de imagen
  POST /recommend   - Recomendación de outfits

Desplegado en Cloud Run como servicio privado.
Autenticación: Firebase ID Token (Bearer)
"""

import logging
import os
from io import BytesIO
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from PIL import Image

from auth import verify_firebase_token, security
from model.loader import ModelLoader
from model.caracteristicas import extraer, extraer_features_array
from model.reglas import recomendar
from storage import storage_manager

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s"}'
)
logger = logging.getLogger(__name__)

# Inicializar aplicación FastAPI
app = FastAPI(
    title="Viste Inteligente - ML Service",
    description="Servicio de clasificación y recomendación de outfits",
    version="1.0.0"
)

# Cargar modelo al inicio
model_loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    """Carga el modelo y inicializa servicios al arrancar."""
    try:
        logger.info("🚀 Iniciando servicio ML...")
        model_loader.load()
        logger.info(f"✅ Modelo cargado: {model_loader.get_info()}")
        
        # Inicializar Cloud Storage
        storage_manager.initialize()
        logger.info("✅ Cloud Storage conectado")
        
    except Exception as e:
        logger.error(f"❌ Error crítico al iniciar: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """
    Health check para Cloud Run.
    Debe responder en < 2 segundos.
    No requiere autenticación.
    """
    try:
        model_info = model_loader.get_info()
        
        return {
            "status": "healthy",
            "service": "viste-inteligente-ml",
            "version": model_info["version"],
            "model_loaded": model_info["loaded"],
            "model_type": model_info["type"],
            "storage_connected": storage_manager._initialized,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/classify")
async def classify_image(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    file: UploadFile = File(..., description="Imagen de la prenda")
):
    """
    Clasifica una imagen y devuelve sus características.
    
    Args:
        file: Imagen subida (JPEG/PNG)
        
    Returns:
        {
            "tipo": str,
            "color": str,
            "ocasiones": [str],
            "patron": str,
            "mezclilla": bool,
            "color_detallado": {nombre, tono, saturacion, brillo, neutro, fuerte}
        }
    """
    # Verificar autenticación
    user_info = await verify_firebase_token(request, credentials)
    
    try:
        # Validar archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer bytes de la imagen
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        
        # Validar tamaño (máximo 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (máx 10MB)")
        
        logger.info(f"Clasificando imagen: {file.filename}, tamaño: {len(contents)} bytes")
        
        # Extraer características
        features_dict = extraer(contents)
        features_array = extraer_features_array(contents).reshape(1, -1)
        
        # Predecir con el modelo
        predictions = model_loader.predict(features_array)
        
        # El modelo puede devolver múltiples clases (tipo, ocasión, patrón)
        # Asumimos que el modelo tiene un pipeline o estructura específica
        # Adaptar según la estructura real de tu modelo
        
        # Por ahora, asumimos que el modelo predice tipo, ocasión y patrón
        # Si tu modelo es diferente, ajusta esta parte
        
        # Ejemplo: si el modelo devuelve un array con [tipo_idx, ocasion_idx, patron_idx]
        # O si tienes modelos separados, llámalos individualmente
        
        # TEMPORAL: Implementación básica - ajustar según tu modelo real
        tipo_pred = int(predictions[0]) if hasattr(predictions, '__len__') else int(predictions)
        
        # Mapear índices a nombres (ajusta según tus clases reales)
        CLASES_TIPO = ["superior", "inferior", "completo", "calzado", "otro"]
        OCASIONES = ["casual", "business_casual", "formal", "deportivo", "fiesta"]
        PATRONES = ["liso", "rayas", "cuadros", "floral", "lunares", "animal print", "gráfico / logo", "otro estampado"]
        
        # Si el modelo tiene predict_proba, usarlo para obtener probabilidades
        probas = model_loader.predict_proba(features_array)
        
        # Determinar tipo (asumiendo que es la primera predicción)
        tipo_idx = tipo_pred % len(CLASES_TIPO)  # Fallback seguro
        tipo = CLASES_TIPO[tipo_idx]
        
        # Para ocasiones y patrones, necesitarías modelos separados o un modelo multi-output
        # Por ahora, devolvemos valores por defecto basados en el tipo
        # TODO: Implementar predicción real de ocasiones y patrones
        
        ocasiones_probables = _predecir_ocasiones(tipo, features_dict)
        patron = _predecir_patron(features_dict)
        
        # Detectar mezclilla (basado en color y textura)
        mezclilla = _detectar_mezclilla(features_dict)
        
        result = {
            "tipo": tipo,
            "color": features_dict["color_nombre"],
            "ocasiones": ocasiones_probables,
            "patron": patron,
            "mezclilla": mezclilla,
            "color_detallado": {
                "nombre": features_dict["color_nombre"],
                "tono": float(features_dict["dom_tono"]),
                "saturacion": float(features_dict["dom_saturacion"]),
                "brillo": float(features_dict["dom_brillo"]),
                "neutro": bool(features_dict["es_neutro"]),
                "fuerte": bool(features_dict["es_fuerte"]),
            },
            "modelo_version": model_loader.version,
        }
        
        logger.info(f"✅ Clasificación completada: tipo={tipo}, color={result['color']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al clasificar imagen: {str(e)}")


def _predecir_ocasiones(tipo: str, features: dict) -> List[str]:
    """
    Predice las ocasiones apropiadas para una prenda.
    TODO: Implementar con modelo real o reglas heurísticas.
    """
    # Por ahora, usar reglas simples basadas en el tipo
    # En producción, usar un modelo entrenado específicamente para ocasiones
    
    ocasiones_base = {
        "superior": ["casual", "business_casual"],
        "inferior": ["casual", "business_casual", "deportivo"],
        "completo": ["casual", "formal", "fiesta"],
        "calzado": ["casual", "deportivo"],
        "otro": ["casual"],
    }
    
    # Ajustar según características
    ocasiones = ocasiones_base.get(tipo, ["casual"])
    
    # Si es neutro y formal, agregar business_casual
    if features.get("es_neutro") and tipo in ["superior", "inferior"]:
        if "business_casual" not in ocasiones:
            ocasiones.append("business_casual")
    
    return ocasiones


def _predecir_patron(features: dict) -> str:
    """
    Predice el patrón de la prenda.
    TODO: Implementar con modelo real o análisis de textura.
    """
    # Por ahora, asumir liso (la mayoría de las prendas son lisas)
    # En producción, usar análisis de textura o modelo entrenado
    
    densidad = features.get("densidad_bordes", 0)
    
    # Heurística simple: alta densidad de bordes podría indicar estampado
    if densidad > 0.3:
        return "otro estampado"
    
    return "liso"


def _detectar_mezclilla(features: dict) -> bool:
    """
    Detecta si la prenda es de mezclilla basado en color y características.
    """
    color = features.get("color_nombre", "").lower()
    
    # La mezclilla típicamente es azul o azul marino
    es_azul = color in ["azul", "azul marino", "celeste"]
    
    # Además, la mezclilla tiene cierta textura (densidad de bordes media-alta)
    densidad = features.get("densidad_bordes", 0)
    
    return es_azul and densidad > 0.15


@app.post("/classify")
async def classify_image(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    file: UploadFile = File(..., description="Imagen de la prenda")
):
    user_info = await verify_firebase_token(request, credentials)
    
    try:
        # 1. Leer y validar imagen
        contents = await file.read()
        if not contents: raise HTTPException(status_code=400, detail="Archivo vacío")
        
        # 2. Extraer features manuales
        features_manual = extraer_features_array(contents, input_type="Manual")
        
        # 3. Predecir TIPO
        tipo_idx_pred = model_loader.predict_tipo(features_manual)[0]
        
        # Mapeo de índices a nombres (asegúrate que coincidan con CLASES en config.py)
        CLASES_TIPO = ["superior", "inferior", "completo", "calzado", "otro"]
        tipo_nombre = CLASES_TIPO[int(tipo_idx_pred)]
        
        # 4. Predecir OCASIONES (si el modelo lo tiene)
        ocasiones_predichas = []
        if 'ocasion' in model_loader.model_data and model_loader.model_data['ocasion']:
            # El modelo de ocasión espera: Features Manuales + OneHot del Tipo
            onehot_tipo = get_onehot_tipo(int(tipo_idx_pred))
            input_ocasion = np.hstack([features_manual, onehot_tipo])
            
            # Obtener predicciones binarias (0 o 1) para cada ocasión
            pred_oc_raw = model_loader.predict_ocasion(input_ocasion)[0]
            
            # Obtener nombres de ocasiones y umbrales si existen
            modelo_oc_data = model_loader.model_data['ocasion']
            ocasiones_nombres = modelo_oc_data.get('ocasiones', [])
            umbrales = modelo_oc_data.get('umbrales', {})
            
            for i, nombre_interno in enumerate(ocasiones_nombres):
                # Usar umbral si existe, sino 0.5 por defecto
                umbral = umbrales.get(nombre_interno, 0.5)
                if pred_oc_raw[i] >= umbral:
                    # Mapear nombre interno (ej: 'oc_casual') a nombre legible
                    nombre_legible = nombre_interno.replace('oc_', '')
                    ocasiones_predichas.append(nombre_legible)

        # 5. Obtener metadatos de color y patrón (heurística por ahora)
        features_dict = extraer(contents) # Diccionario completo
        
        result = {
            "tipo": tipo_nombre,
            "color": features_dict["color_nombre"],
            "ocasiones": ocasiones_predichas if ocasiones_predichas else ["casual"],
            "patron": _predecir_patron(features_dict), # Función heurística anterior
            "mezclilla": _detectar_mezclilla(features_dict),
            "color_detallado": {
                "nombre": features_dict["color_nombre"],
                "tono": float(features_dict["dom_tono"]),
                "saturacion": float(features_dict["dom_saturacion"]),
                "brillo": float(features_dict["dom_brillo"]),
                "neutro": bool(features_dict["es_neutro"]),
                "fuerte": bool(features_dict["es_fuerte"]),
            },
            "modelo_version": model_loader.version,
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")

@app.post("/recommend")
async def recommend_outfits(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_id: str = None,
    ocasion: Optional[str] = None,
    color: Optional[str] = None,
    top: int = 5,
    incluir_closet: bool = True,
    prendas_extra: Optional[List[dict]] = None,
):
    """
    Recomienda outfits basados en el closet del usuario y filtros.
    
    Args:
        user_id: UID del usuario en Firebase
        occasion: Filtro de ocasión (casual, business_casual, formal, deportivo, fiesta)
        color: Filtro de color preferido
        top: Número de outfits a devolver (default: 5)
        incluir_closet: Si incluye prendas del closet del usuario (default: True)
        prendas_extra: Lista adicional de prendas para considerar
        
    Returns:
        {
            "outfits": [
                {
                    "calificacion": float,
                    "piezas": [
                        {
                            "tipo": str,
                            "color": str,
                            "nombre": str,
                            "imagen_url": str
                        }
                    ],
                    "razones": [str]
                }
            ],
            "modelo_version": str,
            "prendas_analizadas": int
        }
    """
    # Verificar autenticación
    user_info = await verify_firebase_token(request, credentials)
    
    # Usar el UID del token si no se proporcionó user_id
    if not user_id:
        user_id = user_info["uid"]
    
    # Validar parámetros
    if top < 1 or top > 20:
        raise HTTPException(status_code=400, detail="El parámetro 'top' debe estar entre 1 y 20")
    
    OCASIONES_VALIDAS = ["casual", "business_casual", "formal", "deportivo", "fiesta"]
    if ocasion and ocasion not in OCASIONES_VALIDAS:
        raise HTTPException(
            status_code=400, 
            detail=f"Ocasion inválida. Valores permitidos: {OCASIONES_VALIDAS}"
        )
    
    try:
        logger.info(f"Generando recomendaciones para usuario {user_id}, ocasión: {ocasion}")
        
        # Obtener prendas del closet del usuario
        prendas = []
        
        if incluir_closet:
            try:
                prendas_closet = storage_manager.get_user_closet(user_id)
                prendas.extend(prendas_closet)
                logger.info(f"📦 Cargadas {len(prendas_closet)} prendas del closet")
            except Exception as e:
                logger.warning(f"No se pudo cargar el closet: {str(e)}")
        
        # Agregar prendas extra si se proporcionaron
        if prendas_extra:
            for prenda in prendas_extra:
                # Validar estructura mínima de la prenda
                if "tipo" not in prenda or "color" not in prenda:
                    logger.warning(f"Prenda extra inválida: {prenda}")
                    continue
                
                # Asegurar que tenga estructura completa
                prenda_completa = {
                    "tipo": prenda["tipo"],
                    "color": {
                        "nombre": prenda.get("color", {}).get("nombre", "desconocido") if isinstance(prenda.get("color"), dict) else prenda.get("color", "desconocido"),
                        "tono": prenda.get("color", {}).get("tono", 0) if isinstance(prenda.get("color"), dict) else 0,
                        "saturacion": prenda.get("color", {}).get("saturacion", 0.5) if isinstance(prenda.get("color"), dict) else 0.5,
                        "brillo": prenda.get("color", {}).get("brillo", 0.5) if isinstance(prenda.get("color"), dict) else 0.5,
                        "neutro": prenda.get("color", {}).get("neutro", False) if isinstance(prenda.get("color"), dict) else False,
                        "fuerte": prenda.get("color", {}).get("fuerte", False) if isinstance(prenda.get("color"), dict) else False,
                    },
                    "patron": prenda.get("patron", "liso"),
                    "mezclilla": prenda.get("mezclilla", False),
                    "ocasiones": prenda.get("ocasiones", ["casual"]),
                    "nombre": prenda.get("nombre", "Prenda"),
                    "imagen_url": prenda.get("imagen_url", ""),
                }
                prendas.append(prenda_completa)
        
        if not prendas:
            raise HTTPException(
                status_code=404, 
                detail="No hay prendas disponibles. Agrega prendas a tu closet primero."
            )
        
        # Filtrar por color si se especificó
        if color:
            prendas_filtradas = [
                p for p in prendas 
                if p.get("color", {}).get("nombre", "").lower() == color.lower()
                or (isinstance(p.get("color"), str) and p.get("color", "").lower() == color.lower())
            ]
            if prendas_filtradas:
                prendas = prendas_filtradas
                logger.info(f"Filtradas {len(prendas)} prendas por color: {color}")
        
        logger.info(f"Analizando {len(prendas)} prendas para recomendación")
        
        # Generar recomendaciones usando el motor de reglas
        resultados = recomendar(prendas, ocasion=ocasion, top=top)
        
        # Formatear respuesta
        outfits = []
        for resultado in resultados:
            piezas_formateadas = []
            for pieza in resultado["piezas"]:
                pieza_fmt = {
                    "tipo": pieza.get("tipo", "desconocido"),
                    "color": pieza.get("color", {}).get("nombre", "desconocido") if isinstance(pieza.get("color"), dict) else pieza.get("color", "desconocido"),
                    "nombre": pieza.get("nombre", "Prenda"),
                    "imagen_url": pieza.get("imagen_url", ""),
                }
                piezas_formateadas.append(pieza_fmt)
            
            outfit = {
                "calificacion": resultado["calificacion"],
                "piezas": piezas_formateadas,
                "razones": resultado["razones"],
            }
            outfits.append(outfit)
        
        response = {
            "outfits": outfits,
            "modelo_version": model_loader.version,
            "prendas_analizadas": len(prendas),
        }
        
        logger.info(f"✅ Recomendaciones generadas: {len(outfits)} outfits")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en recomendación: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    workers = int(os.getenv("UVICORN_WORKERS", 2))
    
    logger.info(f"Iniciando servidor en puerto {port} con {workers} workers")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
    )