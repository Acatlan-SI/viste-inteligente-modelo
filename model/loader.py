"""
loader.py
---------
Carga el modelo .joblib específico de Viste Inteligente.
El modelo es un diccionario con claves 'tipo', 'ocasion' e 'entrada'.
"""

import logging
import os
import joblib

logger = logging.getLogger(__name__)


class ModelLoader:
    """Carga y gestiona el modelo de ML estructurado."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv("MODELS_PATH", "models/modelo.joblib")
        self.model_data = None
        self.version = os.getenv("MODEL_VERSION", "v1")
        self._loaded = False

    def load(self):
        """Carga el modelo desde disco."""
        try:
            logger.info(f"Cargando modelo desde: {self.model_path}")
            self.model_data = joblib.load(self.model_path)
            self._loaded = True
            
            # Validar estructura esperada
            if not isinstance(self.model_data, dict):
                raise ValueError("El modelo cargado no es un diccionario como se esperaba.")
            
            required_keys = ['tipo', 'entrada']
            for key in required_keys:
                if key not in self.model_data:
                    raise ValueError(f"Falta la clave '{key}' en el modelo cargado.")

            logger.info(f"✅ Modelo cargado. Tipo de entrada: {self.model_data.get('entrada')}")
            return self.model_data
            
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            raise RuntimeError(f"Error al cargar modelo: {str(e)}")

    def predict_tipo(self, features_array):
        """Predice el tipo de prenda usando el sub-modelo 'tipo'."""
        if not self._loaded:
            raise RuntimeError("Modelo no cargado.")
        
        try:
            modelo_tipo = self.model_data['tipo']['modelo']
            return modelo_tipo.predict(features_array)
        except Exception as e:
            logger.error(f"Error en predicción de tipo: {str(e)}")
            raise

    def predict_proba_tipo(self, features_array):
        """Devuelve probabilidades para el tipo de prenda."""
        if not self._loaded:
            raise RuntimeError("Modelo no cargado.")
        
        try:
            modelo_tipo = self.model_data['tipo']['modelo']
            if hasattr(modelo_tipo, 'predict_proba'):
                return modelo_tipo.predict_proba(features_array)
            return None
        except Exception as e:
            logger.error(f"Error en predict_proba tipo: {str(e)}")
            raise

    def predict_ocasion(self, features_array_with_onehot):
        """
        Predice las ocasiones. 
        NOTA: Este modelo espera las features + one-hot encoding del tipo.
        """
        if not self._loaded:
            raise RuntimeError("Modelo no cargado.")
        
        if 'ocasion' not in self.model_data or self.model_data['ocasion'] is None:
            return None

        try:
            modelo_ocasion = self.model_data['ocasion']['modelo']
            return modelo_ocasion.predict(features_array_with_onehot)
        except Exception as e:
            logger.error(f"Error en predicción de ocasión: {str(e)}")
            raise

    @property
    def is_loaded(self):
        return self._loaded

    def get_info(self):
        return {
            "version": self.version,
            "path": self.model_path,
            "loaded": self._loaded,
            "input_type": self.model_data.get('entrada') if self.model_data else None,
        }