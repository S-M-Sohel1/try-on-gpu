import base64
import io
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pillow_avif
from typing import Optional

from stage_b.mask_generator import MaskGenerator
from stage_b.catvton_runner import CatVTONRunner

app = FastAPI(title="Fabric Try-On GPU Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferRequest(BaseModel):
    person_image: str   # base64 encoded image
    garment_image: str  # base64 encoded product photo (white background, flat lay)
    garment_category: str       # "upper", "lower", or "overall"
    garment_type: Optional[str] = None  # e.g. "shirt", "t-shirt", "punjabi", "kurta", "jeans"

class InferResponse(BaseModel):
    output_image: str  # base64 encoded result
    model_used: str

mask_generator = None
catvton_runner = None

@app.on_event("startup")
async def startup_event():
    global mask_generator, catvton_runner
    print("Initializing models... This may take a while.")

    # CatVTONRunner downloads the HF repo and exposes repo_path
    catvton_runner = CatVTONRunner()

    # MaskGenerator uses repo_path to find SCHP + DensePose checkpoints
    mask_generator = MaskGenerator(repo_path=catvton_runner.repo_path)

    print("All models initialized successfully.")

def decode_image(b64_str: str) -> Image.Image:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    if request.garment_category not in ["upper", "lower", "overall"]:
        raise HTTPException(status_code=400, detail="garment_category must be 'upper', 'lower', or 'overall'.")

    print(f"Starting inference | category={request.garment_category} | type={request.garment_type}")
    start_time = time.time()

    person_img = decode_image(request.person_image)
    garment_img = decode_image(request.garment_image)

    # Generate cloth-agnostic mask using CatVTON's AutoMasker
    mask = mask_generator.generate_mask(
        person_image=person_img,
        category=request.garment_category,
        garment_type=request.garment_type
    )

    # Run CatVTON inpainting
    output_image = catvton_runner.run(
        person_image=person_img,
        garment_image=garment_img,
        mask=mask
    )

    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    return InferResponse(output_image=encode_image(output_image), model_used="CatVTON+AutoMasker")

@app.get("/health")
def health_check():
    return {"status": "ok"}
