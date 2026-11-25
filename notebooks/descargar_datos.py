"""
Script para descargar datos de entrenamiento desde Cloud Storage
Ejecutar desde la RAÍZ del proyecto: python notebooks/descargar_datos.py
"""

import os
import sys
from pathlib import Path
from google.cloud import storage
import pandas as pd
from tqdm import tqdm

# ============================================
# CONFIGURACIÓN - AJUSTA ESTOS VALORES
# ============================================
BUCKET_NAME = "viste-inteligente-bucket"
GCS_PREFIX = "datos"           # Carpeta en el bucket
LOCAL_DIR = Path("datos")      # 🆕 Cambiado a 'datos' para coincidir con tu notebook
CSV_FILENAME = "dataset_limpio.csv"  # 🆕 Nombre exacto de tu CSV
IMG_SUBFOLDER = "fotos"        # 🆕 Cambiado a 'fotos' para coincidir con tu notebook

# ============================================
# NO MODIFICAR DE AQUÍ EN ADELANTE
# ============================================

def descargar_datos_gcs():
    """Descarga CSV e imágenes desde Cloud Storage"""
    
    # Resolver ruta absoluta basada en la ubicación del proyecto
    project_root = Path(__file__).parent.parent
    data_dir = project_root / LOCAL_DIR
    
    # Crear estructura local
    data_dir.mkdir(exist_ok=True)
    img_dir = data_dir / IMG_SUBFOLDER
    img_dir.mkdir(exist_ok=True)
    
    print(f"📂 Directorio de datos: {data_dir.absolute()}")
    
    # Conectar a GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        print(f"📦 Conectado al bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"❌ Error conectando a GCS: {e}")
        print("💡 Ejecuta: gcloud auth application-default login")
        sys.exit(1)
    
    # 1. Descargar el CSV
    print(f"\n📋 Buscando CSV: {GCS_PREFIX}/{CSV_FILENAME}")
    csv_blob = bucket.blob(f"{GCS_PREFIX}/{CSV_FILENAME}")
    
    if not csv_blob.exists():
        # Intentar buscar cualquier CSV en la carpeta train/
        blobs = list(bucket.list_blobs(prefix=f"{GCS_PREFIX}/"))
        csv_blobs = [b for b in blobs if b.name.endswith('.csv')]
        
        if not csv_blobs:
            print(f"❌ No se encontró ningún CSV en {GCS_PREFIX}/")
            return None
        
        csv_blob = csv_blobs[0]
        print(f"⚠️  CSV no encontrado con nombre esperado. Usando: {csv_blob.name}")
    
    csv_local_path = data_dir / csv_blob.name.split("/")[-1]
    csv_blob.download_to_filename(csv_local_path)
    print(f"✅ CSV descargado: {csv_local_path}")
    
    # Leer y validar CSV
    df = pd.read_csv(csv_local_path)
    print(f"📊 CSV contiene {len(df)} registros")
    print(f"   Columnas: {list(df.columns)}")
    
    # 2. Descargar imágenes
    print(f"\n🖼️  Descargando imágenes de {GCS_PREFIX}/img/ ...")
    img_blobs = [b for b in bucket.list_blobs(prefix=f"{GCS_PREFIX}/img/") 
                 if not b.name.endswith("/")]
    
    print(f"   Encontradas {len(img_blobs)} imágenes")
    
    descargadas = 0
    omitidas = 0
    errores = 0
    
    for blob in tqdm(img_blobs, desc="Descargando"):
        filename = blob.name.split("/")[-1]
        local_path = img_dir / filename
        
        if local_path.exists():
            omitidas += 1
            continue
        
        try:
            blob.download_to_filename(local_path)
            descargadas += 1
        except Exception as e:
            print(f"\n⚠️  Error: {filename} - {e}")
            errores += 1
    
    print(f"\n✅ Resumen:")
    print(f"   Descargadas: {descargadas}")
    print(f"   Omitidas: {omitidas}")
    print(f"   Errores: {errores}")
    print(f"   Total en {img_dir}: {len(list(img_dir.glob('*')))} archivos")
    
    return df


def validar_consistencia(df):
    """Valida que las imágenes del CSV existan localmente"""
    project_root = Path(__file__).parent.parent
    img_dir = project_root / LOCAL_DIR / IMG_SUBFOLDER
    
    print(f"\n🔍 Validando consistencia...")
    
    # Detectar columna de imagen
    col_img = None
    for posible in ['id_imagen', 'filename', 'imagen', 'image', 'file', 'nombre', 'foto']:
        if posible in df.columns:
            col_img = posible
            break
    
    if not col_img:
        print(f"⚠️  No se detectó columna de imágenes.")
        print(f"   Columnas disponibles: {list(df.columns)}")
        return False
    
    print(f"   Columna detectada: '{col_img}'")
    
    existentes = 0
    faltantes = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validando"):
        img_name = str(row[col_img])
        existe = any((img_dir / f"{img_name}{ext}").exists() 
                    for ext in ['', '.jpg', '.jpeg', '.png', '.webp'])
        
        if existe:
            existentes += 1
        else:
            faltantes.append(img_name)
    
    porcentaje = (existentes / len(df) * 100) if len(df) > 0 else 0
    print(f"\n📈 Resultado: {existentes}/{len(df)} ({porcentaje:.1f}%)")
    
    if faltantes:
        print(f"⚠️  Faltantes: {len(faltantes)}")
        for f in faltantes[:5]:
            print(f"   - {f}")
        if len(faltantes) > 5:
            print(f"   ... y {len(faltantes) - 5} más")
    
    return len(faltantes) == 0


if __name__ == "__main__":
    print("=" * 60)
    print("DESCARGA DE DATOS - VISTE INTELIGENTE")
    print("=" * 60)
    
    df = descargar_datos_gcs()
    
    if df is not None:
        validar_consistencia(df)
        print("\n🎉 ¡Datos listos! Ejecuta el notebook de entrenamiento.")