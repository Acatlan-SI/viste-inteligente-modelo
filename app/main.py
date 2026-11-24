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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check para Cloud Run"""
    return {"status": "healthy"}

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