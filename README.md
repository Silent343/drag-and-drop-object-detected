# YOLO Object Detection API — MLOps Portfolio Project

Detector de objetos empaquetado como **API REST con Docker**, desplegado en
Railway. Toma cualquier imagen por HTTP y devuelve las detecciones como JSON
más la imagen anotada en base64.

Es la misma lógica inteligente del `object_detector.py` original (thresholds
por clase, reglas de aspect ratio, límites de área) adaptada para servirse
como microservicio.

## Endpoints

| Método | Ruta      | Descripción                              |
|--------|-----------|------------------------------------------|
| POST   | `/detect` | Subir imagen → JSON con detecciones      |
| GET    | `/health` | Liveness probe (Railway lo usa)          |
| GET    | `/`       | Frontend web (drag & drop + resultados)  |

### Ejemplo de respuesta de `/detect`

```json
{
  "total": 2,
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "class_name_en": "laptop",
      "class_name_es": "Laptop",
      "confidence": 0.9123,
      "bbox": [142, 89, 680, 490],
      "color_hex": "#00b4ff"
    }
  ],
  "annotated_image_b64": "<base64 JPEG con cajas dibujadas>"
}
```

## Correr localmente con Docker

```bash
# Build
docker build -t yolo-api .

# Run
docker run -p 8000:8000 yolo-api

# Probar
curl -X POST http://localhost:8000/detect \
  -F "file=@foto.jpg" | python3 -m json.tool
```

O sin Docker (para desarrollo rápido):
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://localhost:8000
```

## Deploy en Railway

1. Push el proyecto a un repo de GitHub
2. En Railway: **New Project → Deploy from GitHub repo**
3. Railway detecta el `Dockerfile` automáticamente
4. La variable `PORT` la inyecta Railway solo
5. En 3–5 minutos tenés una URL pública como
   `https://yolo-api-production.up.railway.app`

El `railway.toml` ya configura el healthcheck en `/health` para que Railway
sepa cuándo el contenedor está listo (el modelo tarda ~30s en cargar).

## Por qué esto importa en un portfolio

La mayoría de devs de IA saben *usar* modelos. Pocos saben *servirlos*.
Este proyecto demuestra:
- **Docker**: empaquetar un modelo con sus dependencias nativas (OpenCV, YOLO)
- **Producción**: healthcheck, manejo de errores HTTP, límite de tamaño, CORS
- **MLOps real**: el modelo se descarga durante el `docker build` y queda
  baked en la imagen → sin downloads en cold start
- **API design**: response con la imagen anotada en base64, útil para
  cualquier cliente (web, móvil, otro backend)

## Variables de entorno opcionales

| Variable    | Default      | Descripción                        |
|-------------|--------------|------------------------------------|
| `YOLO_MODEL`| `yolov8s.pt` | Modelo YOLO (n/s/m/l/x)            |
| `YOLO_CONF` | `0.40`       | Umbral global de confianza (0–1)   |
| `PORT`      | `8000`       | Puerto (Railway lo inyecta solo)   |
