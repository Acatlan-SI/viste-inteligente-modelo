from fastapi import UploadFile, HTTPException
from app.utils.caracteristicas import FeatureExtractor
from app.utils.color_kmeans import ColorAnalyzer
from app.models.schemas import ImageClassification, ColorAnalysis, ClothingType, Occasion
from typing import List
import io

class ImageService:
    """Servicio para procesamiento de imágenes"""
    
    @staticmethod
    async def analyze_image(image_file: UploadFile) -> ImageClassification:
        """
        Analiza una imagen y extrae clasificación completa
        
        Args:
            image_file: Archivo de imagen subido
            
        Returns:
            Clasificación completa de la imagen
        """
        # Validar archivo
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer bytes
        image_bytes = await image_file.read()
        
        # Extraer características
        features = FeatureExtractor.extract_features(image_bytes)
        
        # Analizar colores
        colors_hex, color_percentages = ColorAnalyzer.get_dominant_colors(image_bytes)
        dominant_color = colors_hex[0] if colors_hex else "#000000"
        
        # Clasificar tipo de prenda
        clothing_type, confidence = FeatureExtractor.classify_clothing_type(features)
        
        # Determinar ocasiones apropiadas (heurística simple)
        occasions = ImageService._determine_occasions(clothing_type, features)
        
        # Crear respuesta
        color_analysis = ColorAnalysis(
            dominant_color=dominant_color,
            color_palette=colors_hex,
            color_percentages=color_percentages
        )
        
        return ImageClassification(
            clothing_type=ClothingType(clothing_type),
            colors=color_analysis,
            occasions=occasions,
            confidence=confidence,
            features=features
        )
    
    @staticmethod
    def _determine_occasions(
        clothing_type: str,
        features: dict
    ) -> List[Occasion]:
        """Determina ocasiones apropiadas para la prenda"""
        occasions = []
        
        # Basado en tipo de prenda
        if clothing_type == 'shirt':
            occasions.extend([Occasion.CASUAL, Occasion.WORK])
        elif clothing_type == 'pants':
            occasions.extend([Occasion.CASUAL, Occasion.FORMAL])
        elif clothing_type == 'jacket':
            occasions.extend([Occasion.CASUAL, Occasion.FORMAL])
        
        # Basado en formalidad (brightness como proxy)
        brightness = features.get('mean_brightness', 128)
        if brightness > 180:
            occasions.append(Occasion.FORMAL)
        elif brightness < 80:
            occasions.append(Occasion.PARTY)
        
        # Eliminar duplicados
        return list(set(occasions))