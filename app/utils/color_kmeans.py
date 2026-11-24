import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from typing import List, Tuple, Dict
import io

class ColorAnalyzer:
    """Análisis de colores dominantes usando K-means"""
    
    @staticmethod
    def get_dominant_colors(
        image_bytes: bytes, 
        n_clusters: int = 3
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Extrae colores dominantes de una imagen usando K-means
        
        Args:
            image_bytes: Bytes de la imagen
            n_clusters: Número de clusters para K-means
            
        Returns:
            Tuple de (lista de colores hex, diccionario de porcentajes)
        """
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = image.resize((150, 150))
        
        # Convertir a array y reshapes para K-means
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Obtener colores de los centros
        colors_rgb = kmeans.cluster_centers_.astype(int)
        
        # Calcular porcentajes
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Convertir a hex
        colors_hex = [ColorAnalyzer._rgb_to_hex(color) for color in colors_rgb]
        
        # Crear diccionario de porcentajes
        color_percentages = {
            color: float(percentage) 
            for color, percentage in zip(colors_hex, percentages)
        }
        
        # Ordenar por porcentaje
        color_percentages = dict(
            sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)
        )
        
        return colors_hex, color_percentages
    
    @staticmethod
    def _rgb_to_hex(rgb: np.ndarray) -> str:
        """Convierte RGB a hexadecimal"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    @staticmethod
    def get_color_name(hex_color: str) -> str:
        """
        Asigna un nombre descriptivo al color hex
        
        Args:
            hex_color: Color en formato hexadecimal
            
        Returns:
            Nombre del color en español
        """
        # Convertir hex a RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Determinar nombre basado en valores RGB
        if r > 200 and g > 200 and b > 200:
            return "blanco"
        elif r < 50 and g < 50 and b < 50:
            return "negro"
        elif r > g and r > b:
            if g > 150:
                return "amarillo"
            elif b > 150:
                return "rosa"
            else:
                return "rojo"
        elif g > r and g > b:
            return "verde"
        elif b > r and b > g:
            return "azul"
        elif r > 150 and g > 100 and b < 100:
            return "naranja"
        else:
            return "gris"