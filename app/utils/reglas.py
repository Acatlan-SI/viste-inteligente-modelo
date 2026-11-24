from typing import List, Dict, Any, Optional
from app.models.schemas import ClothingItem, OutfitRecommendation, Occasion
import uuid

class RecommendationEngine:
    """Motor de reglas para recomendaciones de outfits"""
    
    # Reglas de compatibilidad de colores
    COLOR_COMPATIBILITY = {
        'negro': ['blanco', 'gris', 'rojo', 'azul', 'amarillo'],
        'blanco': ['negro', 'azul', 'rojo', 'verde', 'rosa'],
        'azul': ['blanco', 'negro', 'gris', 'beige', 'amarillo'],
        'rojo': ['negro', 'blanco', 'gris', 'azul'],
        'verde': ['blanco', 'negro', 'beige', 'marrón'],
        'gris': ['negro', 'blanco', 'azul', 'rojo', 'rosa'],
    }
    
    # Reglas por ocasión
    OCCASION_RULES = {
        Occasion.FORMAL: {
            'required_types': ['shirt', 'pants'],
            'preferred_colors': ['negro', 'blanco', 'azul', 'gris'],
            'avoid_colors': ['amarillo', 'naranja']
        },
        Occasion.CASUAL: {
            'required_types': ['shirt', 'pants'],
            'preferred_colors': [],
            'avoid_colors': []
        },
        Occasion.SPORT: {
            'required_types': ['shirt', 'pants'],
            'preferred_colors': ['blanco', 'negro', 'azul'],
            'avoid_colors': []
        },
        Occasion.PARTY: {
            'required_types': ['shirt', 'pants'],
            'preferred_colors': ['negro', 'rojo', 'azul'],
            'avoid_colors': []
        },
        Occasion.WORK: {
            'required_types': ['shirt', 'pants'],
            'preferred_colors': ['azul', 'blanco', 'gris', 'negro'],
            'avoid_colors': ['amarillo', 'naranja', 'rosa']
        }
    }
    
    @classmethod
    def generate_outfits(
        cls,
        available_items: List[ClothingItem],
        occasion: Optional[Occasion] = None,
        max_outfits: int = 5
    ) -> List[OutfitRecommendation]:
        """
        Genera recomendaciones de outfits basado en reglas
        
        Args:
            available_items: Lista de prendas disponibles del usuario
            occasion: Ocasión para filtrar
            max_outfits: Máximo de outfits a generar
            
        Returns:
            Lista de outfits recomendados
        """
        outfits = []
        
        # Agrupar prendas por tipo
        items_by_type = cls._group_by_type(available_items)
        
        # Generar combinaciones
        shirts = items_by_type.get('shirt', [])
        pants = items_by_type.get('pants', [])
        
        for shirt in shirts[:max_outfits]:
            for pants_item in pants[:3]:  # Limitar combinaciones
                outfit = cls._create_outfit(
                    [shirt, pants_item],
                    occasion
                )
                if outfit:
                    outfits.append(outfit)
        
        # Ordenar por score
        outfits.sort(key=lambda x: x.score, reverse=True)
        
        return outfits[:max_outfits]
    
    @classmethod
    def _group_by_type(cls, items: List[ClothingItem]) -> Dict[str, List[ClothingItem]]:
        """Agrupa prendas por tipo"""
        grouped = {}
        for item in items:
            if item.clothing_type.value not in grouped:
                grouped[item.clothing_type.value] = []
            grouped[item.clothing_type.value].append(item)
        return grouped
    
    @classmethod
    def _create_outfit(
        cls,
        items: List[ClothingItem],
        occasion: Optional[Occasion]
    ) -> Optional[OutfitRecommendation]:
        """Crea un outfit y calcula su score"""
        
        # Calcular score de compatibilidad de colores
        color_score = cls._calculate_color_compatibility(items)
        
        # Calcular score de ocasión
        occasion_score = cls._calculate_occasion_match(items, occasion)
        
        # Score total
        total_score = (color_score * 0.6) + (occasion_score * 0.4)
        
        # Generar descripción
        description = cls._generate_description(items, occasion)
        
        return OutfitRecommendation(
            outfit_id=str(uuid.uuid4()),
            items=items,
            score=total_score,
            occasion=occasion or Occasion.CASUAL,
            description=description
        )
    
    @classmethod
    def _calculate_color_compatibility(cls, items: List[ClothingItem]) -> float:
        """Calcula compatibilidad de colores entre prendas"""
        if len(items) < 2:
            return 0.5
        
        # Obtener colores dominantes
        colors = [item.colors.dominant_color for item in items]
        
        # Verificar compatibilidad
        compatible_pairs = 0
        total_pairs = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                total_pairs += 1
                color1 = cls._get_color_name(colors[i])
                color2 = cls._get_color_name(colors[j])
                
                if color2 in cls.COLOR_COMPATIBILITY.get(color1, []):
                    compatible_pairs += 1
        
        return compatible_pairs / total_pairs if total_pairs > 0 else 0.5
    
    @classmethod
    def _calculate_occasion_match(
        cls,
        items: List[ClothingItem],
        occasion: Optional[Occasion]
    ) -> float:
        """Calcula qué tan bien se ajusta el outfit a la ocasión"""
        if not occasion:
            return 0.7
        
        rules = cls.OCCASION_RULES.get(occasion, {})
        
        # Verificar tipos requeridos
        item_types = [item.clothing_type.value for item in items]
        required_types = rules.get('required_types', [])
        
        type_match = all(t in item_types for t in required_types)
        
        # Verificar colores preferidos
        colors = [cls._get_color_name(item.colors.dominant_color) for item in items]
        preferred_colors = rules.get('preferred_colors', [])
        
        if preferred_colors:
            color_match = any(c in preferred_colors for c in colors)
        else:
            color_match = True
        
        return (0.6 if type_match else 0.0) + (0.4 if color_match else 0.0)
    
    @classmethod
    def _get_color_name(cls, hex_color: str) -> str:
        """Obtiene nombre del color desde hex"""
        from app.utils.color_kmeans import ColorAnalyzer
        return ColorAnalyzer.get_color_name(hex_color)
    
    @classmethod
    def _generate_description(
        cls,
        items: List[ClothingItem],
        occasion: Optional[Occasion]
    ) -> str:
        """Genera descripción del outfit"""
        occasion_text = occasion.value if occasion else "casual"
        return f"Outfit {occasion_text} con {len(items)} prendas combinadas"