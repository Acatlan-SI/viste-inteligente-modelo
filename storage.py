"""
storage.py
----------
Integración con Google Cloud Storage para:
1. Leer imágenes y metadatos del closet del usuario
2. Generar URLs firmadas temporales para las imágenes

Bucket: viste-inteligente-bucket
Estructura:
  - users/{uid}/images/{prenda_id}.jpg
  - users/{uid}/metadata/{prenda_id}.json
"""

import json
import logging
import os
from datetime import timedelta
from typing import List, Dict, Optional

from google.cloud import storage

logger = logging.getLogger(__name__)

# Configuración
BUCKET_NAME = os.getenv("GCS_BUCKET", "viste-inteligente-bucket")
URL_EXPIRATION_MINUTES = int(os.getenv("URL_EXPIRATION_MINUTES", "60"))


class CloudStorageManager:
    """Gestiona la lectura de datos desde Cloud Storage."""
    
    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or BUCKET_NAME
        self.client = None
        self.bucket = None
        self._initialized = False
    
    def initialize(self):
        """Inicializa el cliente de Cloud Storage."""
        if self._initialized:
            return
        
        try:
            # En GCP (Cloud Run), usa las credenciales del service account automáticamente.
            # En local, buscará la variable de entorno GOOGLE_APPLICATION_CREDENTIALS.
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            self._initialized = True
            logger.info(f"✅ Conectado a bucket: {self.bucket_name}")
            
        except Exception as e:
            # ⚠️ CAMBIO CLAVE: En lugar de lanzar un error fatal, solo advertimos.
            logger.warning(f"⚠️ No se pudo conectar a Cloud Storage (normal en local sin credenciales): {str(e)}")
            logger.warning("⚠️ El endpoint /recommend no funcionará hasta que se configuren las credenciales de GCP.")
            self._initialized = False
    
    def _ensure_initialized(self):
        """Asegura que el cliente esté inicializado."""
        if not self._initialized:
            self.initialize()
    
    def get_user_closet(self, user_id: str) -> List[Dict]:
        """
        Lee todas las prendas del closet de un usuario desde Cloud Storage.
        
        Args:
            user_id: UID del usuario en Firebase
            
        Returns:
            Lista de dicts con información de cada prenda
        """
        self._ensure_initialized()
        
        try:
            prendas = []
            prefix = f"users/{user_id}/metadata/"
            
            # Listar todos los archivos de metadata del usuario
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if not blob.name.endswith('.json'):
                    continue
                
                try:
                    # Descargar y parsear el metadata
                    content = blob.download_as_text()
                    metadata = json.loads(content)
                    
                    # Agregar URL firmada de la imagen
                    image_blob_name = blob.name.replace('metadata/', 'images/')
                    image_url = self.generate_signed_url(image_blob_name)
                    
                    prenda = {
                        **metadata,
                        "imagen_url": image_url,
                        "user_id": user_id,
                    }
                    prendas.append(prenda)
                    
                except Exception as e:
                    logger.warning(f"Error al leer metadata {blob.name}: {str(e)}")
                    continue
            
            logger.info(f"📦 Cargadas {len(prendas)} prendas para usuario {user_id}")
            return prendas
            
        except Exception as e:
            logger.error(f"Error al leer closet de usuario {user_id}: {str(e)}")
            raise
    
    def generate_signed_url(self, blob_name: str, expiration_minutes: int = None) -> str:
        """
        Genera una URL firmada temporal para acceder a un blob.
        
        Args:
            blob_name: Nombre del objeto en el bucket
            expiration_minutes: Minutos de validez de la URL
            
        Returns:
            URL firmada string
        """
        self._ensure_initialized()
        
        try:
            blob = self.bucket.blob(blob_name)
            expiration = timedelta(minutes=expiration_minutes or URL_EXPIRATION_MINUTES)
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET",
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Error al generar URL firmada para {blob_name}: {str(e)}")
            # Retornar URL placeholder en caso de error
            return f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"
    
    def get_image_bytes(self, user_id: str, prenda_id: str) -> Optional[bytes]:
        """
        Descarga los bytes de una imagen específica.
        
        Args:
            user_id: UID del usuario
            prenda_id: ID de la prenda
            
        Returns:
            Bytes de la imagen o None si no existe
        """
        self._ensure_initialized()
        
        try:
            blob_name = f"users/{user_id}/images/{prenda_id}.jpg"
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                logger.warning(f"Imagen no encontrada: {blob_name}")
                return None
            
            return blob.download_as_bytes()
            
        except Exception as e:
            logger.error(f"Error al descargar imagen {prenda_id}: {str(e)}")
            return None
    
    def get_metadata(self, user_id: str, prenda_id: str) -> Optional[Dict]:
        """
        Lee el metadata de una prenda específica.
        
        Args:
            user_id: UID del usuario
            prenda_id: ID de la prenda
            
        Returns:
            Dict con metadata o None si no existe
        """
        self._ensure_initialized()
        
        try:
            blob_name = f"users/{user_id}/metadata/{prenda_id}.json"
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                logger.warning(f"Metadata no encontrado: {blob_name}")
                return None
            
            content = blob.download_as_text()
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error al leer metadata {prenda_id}: {str(e)}")
            return None


# Instancia singleton para reutilizar
storage_manager = CloudStorageManager()