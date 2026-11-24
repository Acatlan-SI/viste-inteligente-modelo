from PIL import Image
import numpy as np
from typing import Dict, Any, Tuple
import io

class FeatureExtractor:
    """Extracción de características de imágenes de ropa"""
    
    @staticmethod
    def extract_features(image_bytes: bytes) -> Dict[str, Any]:
        """
        Extrae características visuales de una imagen
        
        Args:
            image_bytes: Bytes de la imagen
            
        Returns:
            Diccionario con características extraídas
        """
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Redimensionar para consistencia
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized)
        
        # Características básicas
        features = {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': image.width / image.height,
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'dominant_channels': FeatureExtractor._get_dominant_channels(img_array),
            'texture_complexity': FeatureExtractor._calculate_texture_complexity(img_array),
            'color_variance': FeatureExtractor._calculate_color_variance(img_array)
        }
        
        return features
    
    @staticmethod
    def _get_dominant_channels(img_array: np.ndarray) -> Dict[str, float]:
        """Determina qué canales de color dominan"""
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        
        return {
            'red': float(r_mean),
            'green': float(g_mean),
            'blue': float(b_mean)
        }
    
    @staticmethod
    def _calculate_texture_complexity(img_array: np.ndarray) -> float:
        """Calcula complejidad de textura usando gradientes"""
        gray = np.mean(img_array, axis=2)
        
        # Gradientes simples
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        
        # Magnitud del gradiente
        magnitude = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)
        
        return float(np.mean(magnitude))
    
    @staticmethod
    def _calculate_color_variance(img_array: np.ndarray) -> float:
        """Calcula varianza de color en la imagen"""
        return float(np.var(img_array))
    
    @staticmethod
    def classify_clothing_type(features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Clasifica el tipo de prenda basado en características
        
        Returns:
            Tuple de (tipo_prenda, confianza)
        """
        aspect_ratio = features['aspect_ratio']
        texture = features['texture_complexity']
        
        # Heurísticas simples (reemplazar con modelo ML real)
        if aspect_ratio > 1.5:
            return 'pants', 0.7
        elif aspect_ratio < 0.7:
            return 'shirt', 0.65
        elif texture > 50:
            return 'jacket', 0.6
        else:
            return 'other', 0.5