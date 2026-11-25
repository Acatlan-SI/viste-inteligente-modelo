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
    
    En desarrollo local (ENVIRONMENT != "production"), permite acceso sin token.
    En producción (Cloud Run con ENVIRONMENT=production), valida el token obligatoriamente.
    """
    # Endpoint /health no requiere autenticación
    if request.url.path == "/health":
        return None
    
    # Detectar entorno
    environment = os.getenv("ENVIRONMENT", "development")
    is_production = environment.lower() == "production"
    
    # En producción, SIEMPRE exigir autenticación
    if is_production:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token de autenticación requerido en producción",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = credentials.credentials
        
        try:
            decoded_token = auth.verify_id_token(token)
            user_info = {
                "uid": decoded_token.get("uid"),
                "email": decoded_token.get("email"),
                "email_verified": decoded_token.get("email_verified", False),
            }
            logger.info(f"✅ Usuario autenticado en producción: {user_info['uid']}")
            return user_info
            
        except auth.ExpiredIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="El token ha expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except auth.RevokedIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="El token ha sido revocado",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except auth.InvalidIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Error al verificar token Firebase: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Error de autenticación: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # En desarrollo local, permitir acceso sin token (solo para pruebas)
    else:
        if not credentials:
            logger.warning("⚠️ Modo desarrollo: acceso sin autenticación (NO usar en producción)")
            return {"uid": "local_test_user", "email": "test@local.com"}
        
        # Si hay token en desarrollo, intentar verificarlo (opcional)
        token = credentials.credentials
        try:
            decoded_token = auth.verify_id_token(token)
            return {
                "uid": decoded_token.get("uid"),
                "email": decoded_token.get("email"),
            }
        except Exception as e:
            logger.warning(f"Token inválido en modo desarrollo, usando mock: {str(e)}")
            return {"uid": "local_test_user", "email": "test@local.com"}


def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = None):
    """
    Dependencia de FastAPI para obtener el usuario actual.
    Usa verify_firebase_token internamente.
    """
    # Nota: Esta función se usará como dependencia en los endpoints
    # pero necesita ser async para funcionar con verify_firebase_token
    # Se implementará correctamente en main.py
    pass