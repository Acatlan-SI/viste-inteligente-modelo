from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ML Service - Viste Inteligente"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Google Cloud Storage
    GCS_BUCKET_NAME: str = "viste-inteligente-bucket"
    
    # Configuración de modelos
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp"}
    
    # K-means configuración
    KMEANS_CLUSTERS: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()