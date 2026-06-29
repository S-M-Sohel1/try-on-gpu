import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import base64
import io
import time
import torch

torch.set_float32_matmul_precision("high")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pillow_avif
from typing import Optional, List

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

class EvaluateRequest(BaseModel):
    person_image: str
    garment_image: str
    garment_category: str
    garment_type: Optional[str] = None
    steps_list: List[int] = [20, 30, 40, 50]
    scales_list: List[float] = [1.5, 2.0, 2.5]

class EvaluateResultItem(BaseModel):
    steps: int
    scale: float
    time_taken: float
    output_image: str

class EvaluateResponse(BaseModel):
    results: List[EvaluateResultItem]

mask_generator = None
catvton_runner = None

@app.on_event("startup")
async def startup_event():
    global mask_generator, catvton_runner
    print("Initializing models... This may take a while.")

    # CatVTONRunner downloads CatVTON weights and exposes repo_path
    catvton_runner = CatVTONRunner()

    # MaskGenerator: uses Segformer (primary) for precise clothing masks.
    # repo_path is passed so AutoMasker (SCHP+DensePose) can be used as fallback
    # if Segformer fails to load.
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

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    if request.garment_category not in ["upper", "lower", "overall"]:
        raise HTTPException(status_code=400, detail="garment_category must be 'upper', 'lower', or 'overall'.")

    print(f"Starting evaluation | category={request.garment_category} | type={request.garment_type}")
    
    person_img = decode_image(request.person_image)
    garment_img = decode_image(request.garment_image)

    mask = mask_generator.generate_mask(
        person_image=person_img,
        category=request.garment_category,
        garment_type=request.garment_type
    )

    results = []
    
    if catvton_runner and catvton_runner.pipeline:
        generator = torch.Generator(device=catvton_runner.device).manual_seed(42)
        mask_blurred = catvton_runner.mask_processor.blur(mask, blur_factor=9)
        
        for steps in request.steps_list:
            for scale in request.scales_list:
                print(f"Eval: steps={steps}, scale={scale}")
                start_time = time.time()
                
                result_image = catvton_runner.pipeline(
                    image=person_img.convert("RGB"),
                    condition_image=garment_img.convert("RGB"),
                    mask=mask_blurred,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    height=catvton_runner.height,
                    width=catvton_runner.width,
                    generator=generator
                )[0]
                
                orig_size = person_img.size
                result_image = result_image.resize(orig_size, Image.LANCZOS)
                
                inference_time = time.time() - start_time
                results.append(EvaluateResultItem(
                    steps=steps,
                    scale=scale,
                    time_taken=inference_time,
                    output_image=encode_image(result_image)
                ))
    else:
         raise HTTPException(status_code=500, detail="CatVTON pipeline not loaded.")
    
    return EvaluateResponse(results=results)

@app.get("/health")
def health_check():
    return {"status": "ok"}
