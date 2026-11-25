from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ImageClassification
)
from app.services.image_service import ImageService
from app.services.recommendation_service import RecommendationService
import logging
import os
import joblib
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

modelo_ml = None
metadata_modelo = None


def cargar_modelo_local():
    """Carga el modelo desde el repositorio. La metadata es opcional."""
    global modelo_ml, metadata_modelo

    trained_dir = os.path.join(os.path.dirname(__file__), "models", "trained")

    # === CARGAR MODELO (OBLIGATORIO) ===
    try:
        modelo_path = None
        for ext in ['.joblib', '.pkl', '.bin']:
            candidate = os.path.join(trained_dir, f"outfit_model{ext}")
            if os.path.exists(candidate):
                modelo_path = candidate
                break

        if modelo_path is None:
            archivos = os.listdir(trained_dir) if os.path.exists(trained_dir) else []
            logger.error(f"❌ No se encontró outfit_model.* en {trained_dir}")
            logger.error(f"   Archivos disponibles: {archivos}")
            return False

        data = joblib.load(modelo_path)

        # Manejar dict o modelo directo
        if isinstance(data, dict):
            modelo_ml = data.get('modelo') or data.get('model') or data.get('clf')
        else:
            modelo_ml = data

        logger.info(f"✅ Modelo cargado desde: {modelo_path}")
        logger.info(f"   Tipo: {'dict' if isinstance(data, dict) else 'directo'}")

    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        return False

    # === CARGAR METADATA (OPCIONAL - NO BLOQUEA) ===
    try:
        metadata_path = os.path.join(trained_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_modelo = json.load(f)
            logger.info(f"📊 Metadata cargada: v{metadata_modelo.get('version')}")
            logger.info(f"   Fecha entrenamiento: {metadata_modelo.get('fecha_entrenamiento')}")
            logger.info(f"   Precisión: {metadata_modelo.get('precision')}")
        else:
            logger.warning("⚠️ model_metadata.json no encontrado (no bloqueante)")
            metadata_modelo = {
                "version": "desconocida",
                "fecha_entrenamiento": None,
                "precision": None,
                "descripcion": "Metadata no disponible. Sube model_metadata.json al repo."
            }
    except Exception as e:
        logger.warning(f"⚠️ Error leyendo metadata (no bloqueante): {e}")
        metadata_modelo = {
            "version": "error_lectura",
            "descripcion": str(e)
        }

    return True


# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
image_service = ImageService()
recommendation_service = RecommendationService()


@app.on_event("startup")
async def startup():
    """Se ejecuta al iniciar el servicio"""
    exito = cargar_modelo_local()
    if not exito:
        logger.warning("⚠️ Servicio iniciado sin modelo cargado")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "modelo_cargado": modelo_ml is not None,
        "modelo_version": metadata_modelo.get("version") if metadata_modelo else None
    }


@app.get("/health")
async def health_check():
    """Health check para Cloud Run"""
    return {
        "status": "healthy",
        "modelo_disponible": modelo_ml is not None,
        "modelo_version": metadata_modelo.get("version") if metadata_modelo else None
    }


@app.get("/info-modelo")
async def info_modelo():
    """Endpoint para ver información del modelo actual"""
    if modelo_ml is None:
        return {"error": "Modelo no cargado"}
    return metadata_modelo


@app.post(
    f"{settings.API_V1_STR}/analyze-image",
    response_model=ImageClassification,
    summary="Analizar imagen de prenda",
    description="Extrae características, colores y clasificación de una imagen de ropa"
)
async def analyze_image(image: UploadFile = File(...)):
    """
    Analiza una imagen y devuelve:
    - Tipo de prenda
    - Colores dominantes
    - Ocasiones apropiadas
    - Características visuales
    """
    try:
        result = await image_service.analyze_image(image)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analizando imagen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


@app.post(
    f"{settings.API_V1_STR}/recommend",
    response_model=RecommendationResponse,
    summary="Generar recomendaciones de outfits",
    description="Genera outfits recomendados basado en el guardarropa del usuario y filtros"
)
async def get_recommendations(request: RecommendationRequest):
    """
    Genera recomendaciones de outfits considerando:
    - Prendas disponibles del usuario
    - Ocasión
    - Colores preferidos
    - Items a excluir
    """
    if modelo_ml is None:
        raise HTTPException(status_code=503, detail="Modelo ML no disponible")

    try:
        result = await recommendation_service.get_recommendations(request)
        return result
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )