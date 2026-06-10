import io
import sys
import base64
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "model"))
from predict import predict_image, get_predictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="GlaucomaNet — Glaucoma Detection")
templates = Jinja2Templates(directory=str(HERE / "templates"))

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@app.on_event("startup")
def load_model():
    # Pre-load model at startup so errors surface immediately
    try:
        get_predictor()
        log.info("GlaucomaCNN loaded successfully")
    except Exception as e:
        log.error(f"Model failed to load: {e}")
        log.info("Run model/train.py first to generate model/saved/best_model.pth")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    try:
        meta = get_predictor().meta
    except Exception:
        meta = {}
    return templates.TemplateResponse(request, "index.html", {"meta": meta})


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if Path(file.filename).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)

        # Thumbnail for display
        thumb = img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG")
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        result = predict_image(img)
        result["image_b64"] = thumb_b64
        log.info(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        return result

    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    try:
        return get_predictor().meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
