import base64
import io
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from stage_a.texture_mapper import TextureMapper
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
    person_image: str # base64 encoded image
    fabric_image: str # base64 encoded image
    garment_category: str # "upper", "lower", "overall"
    template_id: Optional[str] = None

class InferResponse(BaseModel):
    output_image: str # base64 encoded image
    model_used: str

texture_mapper = None
mask_generator = None
catvton_runner = None

@app.on_event("startup")
async def startup_event():
    global texture_mapper, mask_generator, catvton_runner
    print("Initializing models... This may take a while.")
    texture_mapper = TextureMapper()
    mask_generator = MaskGenerator()
    catvton_runner = CatVTONRunner()
    print("Models initialized successfully.")

def decode_image(b64_str: str) -> Image.Image:
    try:
        # Remove header if present (e.g., data:image/jpeg;base64,)
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    if request.garment_category not in ["upper", "lower", "overall"]:
        raise HTTPException(status_code=400, detail="Invalid garment_category")

    print(f"Starting inference for category: {request.garment_category}")
    start_time = time.time()
    
    person_img = decode_image(request.person_image)
    fabric_img = decode_image(request.fabric_image)

    # Stage A: Fabric -> Canonical Garment
    canonical_garment = texture_mapper.apply_fabric(
        fabric_image=fabric_img,
        category=request.garment_category,
        template_id=request.template_id
    )

    # Stage B: Auto-Mask Generation
    mask = mask_generator.generate_mask(
        person_image=person_img,
        category=request.garment_category
    )

    # Stage B: CatVTON Inference
    output_image = catvton_runner.run(
        person_image=person_img,
        garment_image=canonical_garment,
        mask=mask
    )
    
    encoded_output = encode_image(output_image)
    
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    return InferResponse(output_image=encoded_output, model_used="CatVTON")

@app.get("/health")
def health_check():
    return {"status": "ok"}
