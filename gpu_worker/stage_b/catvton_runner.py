import torch
from PIL import Image

class CatVTONRunner:
    def __init__(self):
        print("Initializing Real CatVTON Try-On Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

        if self.device == "cuda":
            try:
                # We assume CatVTON repo is cloned and added to sys.path in Colab
                from model.pipeline import CatVTONPipeline
                from huggingface_hub import snapshot_download
                
                print("Loading CatVTON Pipeline... This will take a few minutes on first run.")
                
                repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
                self.pipeline = CatVTONPipeline(
                    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                    attn_ckpt=repo_path,
                    attn_ckpt_version="mix",
                    weight_dtype=torch.float16,
                    use_tf32=True,
                    skip_safety_check=True,
                    device=self.device
                )
            except ImportError as e:
                print(f"Warning: CatVTON imports failed: {e}. Ensure CatVTON is cloned and in sys.path.")
            except Exception as e:
                print(f"Failed to load CatVTON: {e}")
        else:
            print("Warning: CUDA not found. CatVTON requires GPU. Running dummy fallback.")

    def run(self, person_image: Image.Image, garment_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Stage B: Real CatVTON Inference.
        Runs CatVTON to inpaint the clothing realistically.
        """
        if not self.pipeline:
            # Dummy fallback if model didn't load
            person_image = person_image.convert("RGBA")
            garment_image = garment_image.resize(person_image.size).convert("RGBA")
            mask = mask.resize(person_image.size).convert("L")
            result = Image.composite(garment_image, person_image, mask)
            return result.convert("RGB")

        # CatVTON expects images to be standardized
        target_size = (768, 768)
        person_resized = person_image.resize(target_size).convert("RGB")
        garment_resized = garment_image.resize(target_size).convert("RGB")
        mask_resized = mask.resize(target_size).convert("L")

        print("Running CatVTON inference...")
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # Run inference
        result_image = self.pipeline(
            image=person_resized,
            condition_image=garment_resized,
            mask=mask_resized,
            num_inference_steps=50,
            guidance_scale=2.5,
            generator=generator
        )[0]

        # Resize back to original
        return result_image.resize(person_image.size)
