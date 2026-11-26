"""
main.py
-------
Servicio backend de ML para "Viste Inteligente".
"""

import logging
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from PIL import Image

from auth import verify_firebase_token, security
from model.loader import ModelLoader
from model.caracteristicas import extraer, extraer_features_array, get_onehot_tipo
from model.reglas import recomendar
from storage import storage_manager

from pydantic import BaseModel
from typing import List, Optional

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
        
        # Inicializar Cloud Storage (fallará silenciosamente en local si no hay credenciales)
        storage_manager.initialize()
        if storage_manager._initialized:
            logger.info("✅ Cloud Storage conectado")
        else:
            logger.warning("⚠️ Cloud Storage no conectado (modo local o sin credenciales)")
            
    except Exception as e:
        logger.error(f"❌ Error crítico al iniciar: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check para Cloud Run. No requiere autenticación."""
    try:
        model_info = model_loader.get_info()
        return {
            "status": "healthy",
            "service": "viste-inteligente-ml",
            "version": model_info["version"],
            "model_loaded": model_info["loaded"],
            "model_input_type": model_info["input_type"],
            "storage_connected": storage_manager._initialized,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})


@app.post("/classify")
async def classify_image(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    file: UploadFile = File(..., description="Imagen de la prenda")
):
    """Clasifica una imagen y devuelve sus características."""
    
    # 1. Autenticación (con bypass para modo local)
    user_info = await verify_firebase_token(request, credentials)
    
    try:
        # 2. Validar archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (máx 10MB)")
        
        logger.info(f"Clasificando imagen: {file.filename}, tamaño: {len(contents)} bytes")
        
        # 3. Extraer features (1067 dimensiones: 1000 CNN + 67 manuales)
        input_type = model_loader.model_data.get('entrada', 'CNN (1000)')
        features_array = extraer_features_array(contents, input_type=input_type)
        features_dict = extraer(contents) # Para metadatos de color
        
        # 4. Predecir TIPO
        tipo_nombre = str(model_loader.predict_tipo(features_array)[0])

        # Para el modelo de ocasiones, necesitamos el índice de la clase
        CLASES_TIPO = ["superior", "inferior", "completo", "calzado", "otro"]
        try:
            tipo_idx_pred = CLASES_TIPO.index(tipo_nombre)
        except ValueError:
            logger.warning(f"Tipo '{tipo_nombre}' no encontrado en CLASES_TIPO, usando 0")
            tipo_idx_pred = 0
        
        # 5. Predecir OCASIONES (si el modelo lo tiene)
        ocasiones_predichas = ["casual"] # Default
        if 'ocasion' in model_loader.model_data and model_loader.model_data['ocasion'] is not None:
            try:
                # El modelo de ocasión espera: Features (1067) + OneHot del Tipo (5) = 1072
                onehot_tipo = get_onehot_tipo(tipo_idx_pred)
                input_ocasion = np.hstack([features_array, onehot_tipo])
                
                pred_oc_raw = model_loader.predict_ocasion(input_ocasion)[0]
                modelo_oc_data = model_loader.model_data['ocasion']
                ocasiones_nombres = modelo_oc_data.get('ocasiones', [])
                umbrales = modelo_oc_data.get('umbrales', {})
                
                ocasiones_predichas = []
                for i, nombre_interno in enumerate(ocasiones_nombres):
                    umbral = umbrales.get(nombre_interno, 0.5)
                    if pred_oc_raw[i] >= umbral:
                        nombre_legible = nombre_interno.replace('oc_', '')
                        ocasiones_predichas.append(nombre_legible)
                
                if not ocasiones_predichas:
                    ocasiones_predichas = ["casual"]
                    
            except Exception as e:
                logger.warning(f"Error al predecir ocasiones, usando default: {str(e)}")

        # 6. Heurísticas para patrón y mezclilla
        densidad = features_dict.get("densidad_bordes", 0)
        patron = "otro estampado" if densidad > 0.3 else "liso"
        
        color_nombre = features_dict["color_nombre"].lower()
        mezclilla = color_nombre in ["azul", "azul marino", "celeste"] and densidad > 0.15
        
        # 7. Construir respuesta
        result = {
            "tipo": tipo_nombre,
            "color": color_nombre,
            "ocasiones": ocasiones_predichas,
            "patron": patron,
            "mezclilla": mezclilla,
            "color_detallado": {
                "nombre": color_nombre,
                "tono": float(features_dict["dom_tono"]),
                "saturacion": float(features_dict["dom_saturacion"]),
                "brillo": float(features_dict["dom_brillo"]),
                "neutro": bool(features_dict["es_neutro"]),
                "fuerte": bool(features_dict["es_fuerte"]),
            },
            "modelo_version": model_loader.version,
        }
        
        logger.info(f"✅ Clasificación completada: tipo={tipo_nombre}, color={color_nombre}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al clasificar imagen: {str(e)}")


# --- MODELO PYDANTIC PARA EL REQUEST ---
class ColorData(BaseModel):
    nombre: str
    tono: Optional[float] = 0.0
    saturacion: Optional[float] = 0.5
    brillo: Optional[float] = 0.5
    neutro: Optional[bool] = False
    fuerte: Optional[bool] = False

class PrendaExtra(BaseModel):
    tipo: str
    color: ColorData
    patron: Optional[str] = "liso"
    mezclilla: Optional[bool] = False
    ocasiones: Optional[List[str]] = ["casual"]
    nombre: Optional[str] = "Prenda"
    imagen_url: Optional[str] = ""

class RecommendRequest(BaseModel):
    user_id: str
    ocasion: Optional[str] = None
    color: Optional[str] = None
    top: int = 5
    incluir_closet: bool = True
    prendas_extra: Optional[List[PrendaExtra]] = []


# --- ENDPOINT ACTUALIZADO ---
@app.post("/recommend")
async def recommend_outfits(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    payload: RecommendRequest = None,  # ← AQUÍ ESTÁ LA CLAVE: recibimos el JSON como un objeto
):
    """Recomienda outfits basados en el closet del usuario y filtros."""
    
    # Si no se envió payload (por compatibilidad con query params antiguos), usar defaults
    if payload is None:
        payload = RecommendRequest(user_id="unknown")

    user_info = await verify_firebase_token(request, credentials)
    
    # Usar el UID del token si no se proporcionó o no coincide (seguridad)
    user_id = payload.user_id or user_info.get("uid", "local_test_user")
    
    if payload.top < 1 or payload.top > 20:
        raise HTTPException(status_code=400, detail="El parámetro 'top' debe estar entre 1 y 20")
    
    OCASIONES_VALIDAS = ["casual", "business_casual", "formal", "deportivo", "fiesta"]
    if payload.ocasion and payload.ocasion not in OCASIONES_VALIDAS:
        raise HTTPException(status_code=400, detail=f"Ocasion inválida. Valores: {OCASIONES_VALIDAS}")
    
    try:
        logger.info(f"Generando recomendaciones para usuario {user_id}, ocasión: {payload.ocasion}")
        prendas = []
        
        # 1. Cargar desde Cloud Storage si se solicita
        if payload.incluir_closet:
            try:
                prendas_closet = storage_manager.get_user_closet(user_id)
                prendas.extend(prendas_closet)
            except Exception as e:
                logger.warning(f"No se pudo cargar el closet (¿modo local o sin permisos?): {str(e)}")
        
        # 2. Agregar prendas extra enviadas por el frontend
        if payload.prendas_extra:
            for p in payload.prendas_extra:
                prenda_completa = {
                    "tipo": p.tipo,
                    "color": p.color.dict() if hasattr(p.color, 'dict') else p.color,
                    "patron": p.patron,
                    "mezclilla": p.mezclilla,
                    "ocasiones": p.ocasiones,
                    "nombre": p.nombre,
                    "imagen_url": p.imagen_url,
                }
                prendas.append(prenda_completa)
        
        if not prendas:
            raise HTTPException(status_code=404, detail="No hay prendas disponibles. Agrega prendas a tu clóset o sube fotos.")
        
        # 3. Filtrar por color si se especificó
        if payload.color:
            prendas = [
                p for p in prendas 
                if p.get("color", {}).get("nombre", "").lower() == payload.color.lower()
            ]
        
        # 4. Generar recomendaciones
        resultados = recomendar(prendas, ocasion=payload.ocasion, top=payload.top)
        
        # 5. Formatear respuesta (manteniendo estructura completa)
        outfits = []
        for resultado in resultados:
            piezas_formateadas = []
            for pieza in resultado["piezas"]:
                # Color: siempre como diccionario para consistencia
                color_data = pieza.get("color", {})
                if isinstance(color_data, dict):
                    color_dict = {
                        "nombre": color_data.get("nombre", "desconocido"),
                        "tono": color_data.get("tono", 0),
                        "saturacion": color_data.get("saturacion", 0.5),
                        "brillo": color_data.get("brillo", 0.5),
                        "neutro": color_data.get("neutro", False),
                        "fuerte": color_data.get("fuerte", False),
                    }
                else:
                    # Si por alguna razón viene como string, convertirlo
                    color_dict = {"nombre": str(color_data)}
                
                pieza_fmt = {
                    "tipo": pieza.get("tipo", "desconocido"),
                    "color": color_dict,  # ✅ Siempre diccionario
                    "nombre": pieza.get("nombre", "Prenda"),
                    "imagen_url": pieza.get("imagen_url", ""),  # URL firmada de GCS
                }
                piezas_formateadas.append(pieza_fmt)
            
            outfits.append({
                "calificacion": resultado["calificacion"],
                "piezas": piezas_formateadas,
                "razones": resultado["razones"],
            })
        
        return {
            "outfits": outfits,
            "modelo_version": model_loader.version,
            "prendas_analizadas": len(prendas),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en recomendación: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {str(e)}")