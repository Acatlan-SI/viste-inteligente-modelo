# Viste Inteligente - Backend del modelo

Servicio backend de clasificación de prendas y recomendación de outfits. Está construido con FastAPI y expone la API del modelo; esta documentación cubre únicamente este backend.

El proyecto está desplegado en Google Cloud Run: [viste-inteligente.cloud.run](https://viste-inteligente.cloud.run).

## Funcionalidades

- Clasificación de imágenes de prendas.
- Extracción de características visuales: MobileNetV2 ejecutada con ONNX, histogramas HSV, forma y color dominante.
- Recomendación y puntuación de outfits según color, estampados y ocasión.
- Lectura opcional del clóset de un usuario desde Google Cloud Storage (GCS).
- Autenticación de Firebase ID tokens en producción.

Los modelos ya entrenados se encuentran en `models/`. El repositorio no incluye el conjunto de datos ni el código original de entrenamiento; la carpeta `entrenamiento/` queda reservada para incorporar y mostrar esos scripts cuando estén disponibles.

## Requisitos

- Windows, PowerShell y Python 3.12.
- Git para clonar el repositorio.
- Para consultar un clóset desde GCS en local: credenciales de Google Cloud con acceso al bucket configurado.
- Para probar autenticación de producción: un Firebase ID token válido. En el entorno local de desarrollo la API permite peticiones sin token.

## Instalación y ejecución local

Desde PowerShell, en la carpeta raíz del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El primer comando crea el entorno virtual `.venv`; activarlo hace que Python y `pip` usen ese entorno. `pip install -r requirements.txt` instala las dependencias necesarias. Crea `.env` únicamente en tu equipo para configurar valores locales y credenciales, si hacen falta.

Inicia la API:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Al arrancar, el servicio carga `models/modelo.joblib` y prepara la conexión a GCS. Si no hay credenciales de GCP, el servicio sigue funcionando sin acceso al closet remoto; se pueden enviar prendas directamente a `/recommend`.

- API local: <http://127.0.0.1:8000>
- Documentación interactiva de Swagger: <http://127.0.0.1:8000/docs>
- Esquema OpenAPI: <http://127.0.0.1:8000/openapi.json>

Detén el proceso con `Ctrl+C`. Para volver a usar el entorno después de cerrar PowerShell, actívalo de nuevo con `.\.venv\Scripts\Activate.ps1`.

### Variables de entorno

`main.py` carga `.env` al iniciar. La plantilla `.env.example` contiene valores no secretos:

| Variable | Uso | Valor local sugerido |
|---|---|---|
| `ENVIRONMENT` | En `production` exige Firebase Bearer token; en desarrollo permite probar sin token. | `development` |
| `MODEL_VERSION` | Versión incluida en las respuestas de la API. | `v1` |
| `MODELS_PATH` | Ruta al artefacto de clasificación. | `models/modelo.joblib` |
| `GCS_BUCKET` | Bucket usado para leer prendas del clóset. | `viste-inteligente-bucket` |
| `URL_EXPIRATION_MINUTES` | Duración de las URLs firmadas de imágenes. | `60` |
| `FIREBASE_SERVICE_ACCOUNT` | Ruta local a una llave de servicio Firebase, si se requiere validación local. | Sin valor |
| `GOOGLE_APPLICATION_CREDENTIALS` | Ruta a credenciales de Google Cloud para GCS local; también se puede configurar fuera de `.env`. | Sin valor |

El archivo `.env` y las llaves de servicio son privados: no se incluyen en el repositorio y no deben compartirse ni subirse a Git. Para acceder a GCS en local, configura credenciales con permisos adecuados; sin ellas, `/recommend` puede funcionar enviando `prendas_extra` y usando `incluir_closet: false`.

## API

Las rutas de clasificación y recomendación usan autenticación Firebase en producción (`ENVIRONMENT=production`). Envía el ID token en el encabezado:

```text
Authorization: Bearer <FIREBASE_ID_TOKEN>
```

En desarrollo local, el backend permite omitir el encabezado. `/health` no requiere autenticación.

### `GET /health`

Comprueba el estado del servicio y devuelve la versión, si el modelo está cargado, el tipo de entrada esperado y el estado de conexión a GCS.

Ejemplo de respuesta:

```json
{
  "status": "healthy",
  "service": "viste-inteligente-ml",
  "version": "v1",
  "model_loaded": true,
  "model_input_type": "CNN (1000)",
  "storage_connected": false
}
```

### `POST /classify`

Recibe una imagen como `multipart/form-data` en el campo `file`. Acepta imágenes de hasta 10 MB. El modelo predice el tipo de prenda; el servicio también extrae color y atributos visuales, y predice ocasiones si el artefacto incluye ese submodelo.

Ejemplo local con `curl.exe`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/classify" -F "file=@.\prenda.jpg;type=image/jpeg"
```

La respuesta incluye:

- `tipo`: tipo de prenda predicho (`superior`, `inferior`, `completo`, `calzado` u `otro`).
- `color` y `color_detallado`: nombre, tono, saturación, brillo y atributos neutro/fuerte.
- `ocasiones`: ocasiones predichas, o `casual` si no hay un submodelo de ocasión disponible.
- `patron` y `mezclilla`: estimaciones mediante reglas heurísticas.
- `modelo_version`: versión configurada del modelo.

Errores habituales: `400` si el archivo no es una imagen, está vacío o supera el límite; `401` si falta o no es válido el token en producción; `500` si ocurre un error durante la clasificación.

### `POST /recommend`

Recibe un JSON para combinar prendas y devolver hasta 20 outfits ordenados por calificación. Puede usar el clóset de GCS, las prendas incluidas en la solicitud o ambos.

| Campo | Tipo | Descripción |
|---|---|---|
| `user_id` | string | UID utilizado para consultar el clóset de GCS. |
| `ocasion` | string o `null` | Filtro: `casual`, `business_casual`, `formal`, `deportivo` o `fiesta`. |
| `color` | string o `null` | Filtro opcional por nombre de color. |
| `top` | entero | Cantidad máxima de resultados; debe estar entre 1 y 20. |
| `incluir_closet` | booleano | Si es `true`, intenta cargar el clóset desde GCS. |
| `prendas_extra` | arreglo | Prendas adicionales enviadas en la solicitud. |

Cada elemento de `prendas_extra` puede incluir `tipo`, `color`, `patron`, `mezclilla`, `ocasiones`, `nombre` e `imagen_url`. `color` es un objeto con `nombre` y, opcionalmente, `tono`, `saturacion`, `brillo`, `neutro` y `fuerte`.

Ejemplo:

```json
{
  "user_id": "uid-del-usuario",
  "ocasion": "casual",
  "color": null,
  "top": 5,
  "incluir_closet": false,
  "prendas_extra": [
    {
      "tipo": "superior",
      "color": {
        "nombre": "azul",
        "tono": 220,
        "saturacion": 0.7,
        "brillo": 0.7,
        "neutro": false,
        "fuerte": true
      },
      "patron": "liso",
      "mezclilla": false,
      "ocasiones": ["casual"],
      "nombre": "Camisa azul",
      "imagen_url": ""
    }
  ]
}
```

La respuesta contiene `outfits` (piezas, calificación de 0 a 10 y razones), `modelo_version` y `prendas_analizadas`. Si no hay prendas disponibles devuelve `404`; los valores inválidos de `top` u `ocasion` devuelven `400`.

## Qué hace cada función

### `main.py` - API FastAPI

- `startup_event`: carga el modelo y trata de inicializar GCS al iniciar la aplicación.
- `health_check`: devuelve el estado del servicio, modelo y almacenamiento.
- `classify_image`: autentica la solicitud, valida la imagen, extrae características, predice tipo y ocasiones, aplica heurísticas para patrón/mezclilla y arma la respuesta.
- `recommend_outfits`: autentica la solicitud, obtiene prendas del clóset y/o del JSON, aplica filtros, genera combinaciones y formatea las recomendaciones.
- `ColorData`, `PrendaExtra` y `RecommendRequest`: modelos Pydantic para validar la estructura de entrada de `/recommend`.

### Autenticación - `auth.py`

- `get_firebase_app`: inicializa Firebase Admin con la credencial definida en `FIREBASE_SERVICE_ACCOUNT` o con credenciales predeterminadas de Google Cloud.
- `verify_firebase_token`: permite acceso local sin token cuando no está en producción y verifica los Firebase ID tokens en producción. `/health` queda exento.
- `get_current_user`: función sin implementación; no está conectada a las rutas actuales.

### Almacenamiento - `storage.py`

- `CloudStorageManager.__init__`: configura el bucket y el estado inicial del cliente.
- `initialize`: crea el cliente de GCS y selecciona el bucket.
- `_ensure_initialized`: inicializa el cliente si aún no está listo.
- `get_user_closet`: lista los metadatos de prendas bajo `users/{uid}/metadata/` y agrega URLs de imagen.
- `generate_signed_url`: genera una URL temporal de lectura para una imagen.
- `get_image_bytes`: descarga los bytes de una imagen por usuario e identificador.
- `get_metadata`: obtiene el JSON de metadatos de una prenda.
- `storage_manager`: instancia compartida de `CloudStorageManager` usada por la API.

### Carga y predicción - `model/loader.py`

- `ModelLoader.__init__`: prepara la ruta del modelo y su versión.
- `load`: lee el archivo Joblib y valida que tenga la estructura requerida.
- `predict_tipo`: predice el tipo de prenda.
- `predict_proba_tipo`: devuelve probabilidades de tipo cuando el estimador las soporta.
- `predict_ocasion`: predice las ocasiones usando el submodelo opcional.
- `is_loaded`: indica si el modelo se cargó.
- `get_info`: devuelve versión, ruta, estado y tipo de entrada.

### Características visuales - `model/caracteristicas.py`

- `_obtener_sesion_cnn`: crea y reutiliza la sesión ONNX de MobileNetV2.
- `_preparar_imagen_cnn`: redimensiona, recorta y normaliza una imagen para la CNN.
- `obtener_vector_cnn`: obtiene el vector de características CNN.
- `abrir`: abre una imagen desde bytes, ruta o instancia PIL y la convierte a RGB.
- `_histograma_hsv`: calcula histogramas normalizados de tono, saturación y brillo.
- `_forma`: calcula un descriptor de orientación de bordes y densidad.
- `extraer`: combina características manuales con metadatos de color.
- `extraer_features_array`: prepara el vector de entrada que requiere el modelo (manual, CNN o combinado).
- `get_onehot_tipo`: codifica el tipo de prenda como vector one-hot para el submodelo de ocasiones.

### Color dominante - `model/color_kmeans.py`

- `kmeans`: agrupa píxeles RGB con k-means.
- `cargar_pixeles`: reduce la imagen y separa los píxeles centrales de los bordes.
- `color_dominante`: calcula la paleta e intenta excluir el color de fondo.
- `describir_color`: convierte RGB a HSV y asigna nombre y atributos al color.
- `rgb_a_hex`: convierte una tupla RGB a su representación hexadecimal.

### Reglas de outfits - `model/reglas.py`

- `estampado`: determina si una prenda no es lisa.
- `color_de`: devuelve los atributos de color y trata la mezclilla como neutra.
- `diferencia_tono`: calcula la distancia angular entre dos tonos.
- `calificar_par`: asigna puntos y razones a una combinación de dos colores.
- `colorimetria`: ajusta la puntuación según la ocasión.
- `escalar`: convierte la puntuación interna a una escala de 0 a 10.
- `calificar_outfit`: califica una combinación de prenda superior, inferior y calzado opcional, considerando contraste y estampados.
- `calificar_completo`: califica una prenda completa (por ejemplo, vestido) con calzado.
- `filtrar_ocasion`: conserva las prendas disponibles para la ocasión seleccionada.
- `recomendar`: genera combinaciones de prendas completas o de prendas superiores/inferiores con calzado, las ordena y devuelve las mejores.

## Estructura del proyecto

```text
.
├── main.py                 # API y endpoints
├── auth.py                 # Verificación de Firebase
├── storage.py              # Integración con Google Cloud Storage
├── model/                  # Extracción de características y reglas
├── models/                 # Artefactos ONNX y Joblib ya entrenados
├── entrenamiento/          # Espacio reservado para scripts de entrenamiento
├── requirements.txt
├── Dockerfile
└── cloudbuild.yaml         # Construcción y despliegue en Cloud Run
```
