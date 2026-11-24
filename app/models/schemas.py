from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ClothingType(str, Enum):
    SHIRT = "shirt"
    PANTS = "pants"
    DRESS = "dress"
    JACKET = "jacket"
    SHOES = "shoes"
    ACCESSORY = "accessory"
    OTHER = "other"

class Occasion(str, Enum):
    CASUAL = "casual"
    FORMAL = "formal"
    SPORT = "sport"
    PARTY = "party"
    WORK = "work"

class ColorAnalysis(BaseModel):
    dominant_color: str
    color_palette: List[str]
    color_percentages: Dict[str, float]

class ImageClassification(BaseModel):
    clothing_type: ClothingType
    colors: ColorAnalysis
    occasions: List[Occasion]
    confidence: float
    features: Dict[str, Any]

class RecommendationRequest(BaseModel):
    user_id: str
    occasion: Optional[Occasion] = None
    preferred_colors: Optional[List[str]] = None
    exclude_items: Optional[List[str]] = None
    max_items: int = Field(default=5, ge=1, le=20)

class ClothingItem(BaseModel):
    item_id: str
    image_url: str
    clothing_type: ClothingType
    colors: ColorAnalysis
    occasions: List[Occasion]

class OutfitRecommendation(BaseModel):
    outfit_id: str
    items: List[ClothingItem]
    score: float
    occasion: Occasion
    description: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[OutfitRecommendation]
    total_found: int