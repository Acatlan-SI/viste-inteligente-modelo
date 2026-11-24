from google.cloud import storage
from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ClothingItem,
    ClothingType,
    ColorAnalysis,
    Occasion
)
from app.utils.reglas import RecommendationEngine
from app.config import settings
from typing import List
import json

class RecommendationService:
    """Servicio para generar recomendaciones de outfits"""
    
    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(settings.GCS_BUCKET_NAME)
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Genera recomendaciones de outfits para un usuario
        
        Args:
            request: Solicitud de recomendación
            
        Returns:
            Respuesta con outfits recomendados
        """
        # Obtener prendas del usuario desde Cloud Storage
        user_items = await self._get_user_items(request.user_id)
        
        # Filtrar elementos excluidos
        if request.exclude_items:
            user_items = [
                item for item in user_items 
                if item.item_id not in request.exclude_items
            ]
        
        # Filtrar por colores preferidos
        if request.preferred_colors:
            user_items = self._filter_by_colors(
                user_items,
                request.preferred_colors
            )
        
        # Generar outfits
        outfits = RecommendationEngine.generate_outfits(
            available_items=user_items,
            occasion=request.occasion,
            max_outfits=request.max_items
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=outfits,
            total_found=len(outfits)
        )
    
    async def _get_user_items(self, user_id: str) -> List[ClothingItem]:
        """
        Obtiene las prendas del usuario desde Cloud Storage
        
        Nota: Esto asume que hay un archivo JSON con metadata de prendas
        por usuario. Ajustar según tu estructura real.
        """
        try:
            blob = self.bucket.blob(f"users/{user_id}/wardrobe.json")
            
            if not blob.exists():
                return []
            
            data = blob.download_as_text()
            items_data = json.loads(data)
            
            # Convertir a objetos ClothingItem
            items = []
            for item_data in items_data:
                item = ClothingItem(
                    item_id=item_data['item_id'],
                    image_url=item_data['image_url'],
                    clothing_type=ClothingType(item_data['clothing_type']),
                    colors=ColorAnalysis(**item_data['colors']),
                    occasions=[Occasion(o) for o in item_data['occasions']]
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            print(f"Error obteniendo items del usuario: {e}")
            return []
    
    def _filter_by_colors(
        self,
        items: List[ClothingItem],
        preferred_colors: List[str]
    ) -> List[ClothingItem]:
        """Filtra prendas por colores preferidos"""
        from app.utils.color_kmeans import ColorAnalyzer
        
        filtered = []
        for item in items:
            color_name = ColorAnalyzer.get_color_name(item.colors.dominant_color)
            if color_name in preferred_colors:
                filtered.append(item)
        
        return filtered