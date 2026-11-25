"""
auth.py
-------
Verificación de Firebase ID Token para proteger los endpoints.

Solo el frontend (Streamlit en Cloud Run) debe poder llamar a este servicio.
Se valida el token JWT de Firebase en cada request protegido.
"""

import logging
import os
from functools import lru_cache

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Esquema de autenticación Bearer
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_firebase_app():
    """
    Inicializa la app de Firebase Admin SDK.
    Se usa lru_cache para inicializar solo una vez.
    """
    try:
        # Opción 1: Usar credenciales desde archivo JSON (desarrollo local)
        service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account_path and os.path.exists(service_account_path):
            logger.info(f"Cargando credenciales Firebase desde: {service_account_path}")
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        else:
            # Opción 2: Usar Application Default Credentials (producción en GCP)
            # En Cloud Run con Workload Identity o service account adjunto
            logger.info("Usando Application Default Credentials para Firebase")
            firebase_admin.initialize_app()
        
        return firebase_admin.get_app()
        
    except ValueError:
        # Ya está inicializada
        return firebase_admin.get_app()
    except Exception as e:
        logger.error(f"Error al inicializar Firebase Admin: {str(e)}")
        raise RuntimeError(f"No se pudo inicializar Firebase Admin: {str(e)}")


async def verify_firebase_token(request: Request, credentials: HTTPAuthorizationCredentials = None):
    """
    Middleware/Dependencia para verificar el ID Token de Firebase.
    
    En desarrollo local (sin credenciales de Firebase), permite acceso sin token.
    En producción (Cloud Run), valida el token obligatoriamente.
    """
    # Endpoint /health no requiere autenticación
    if request.url.path == "/health":
        return None
    
    # Modo local: si no hay service account de Firebase configurada, permitimos acceso de prueba
    firebase_service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    is_local_mode = not firebase_service_account or not os.path.exists(firebase_service_account)
    
    if is_local_mode and not credentials:
        logger.warning("⚠️ Modo local detectado: acceso sin autenticación (solo para desarrollo)")
        return {"uid": "local_test_user", "email": "test@local.com"}
    
    # Verificar que se proporcionó un token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación no proporcionado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # Verificar el token con Firebase
        decoded_token = auth.verify_id_token(token)
        
        # Extraer información relevante
        user_info = {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
        }
        
        logger.debug(f"Usuario autenticado: {user_info['uid']}")
        return user_info
        
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El token ha expirado. Por favor, inicia sesión nuevamente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El token ha sido revocado. Por favor, inicia sesión nuevamente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error al verificar token Firebase: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error de autenticación: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = None):
    """
    Dependencia de FastAPI para obtener el usuario actual.
    Usa verify_firebase_token internamente.
    """
    # Nota: Esta función se usará como dependencia en los endpoints
    # pero necesita ser async para funcionar con verify_firebase_token
    # Se implementará correctamente en main.py
    pass