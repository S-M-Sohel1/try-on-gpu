import torch
from diffusers import VaeImageProcessor
from PIL import Image

class CatVTONRunner:
    def __init__(self):
        print("Initializing Real CatVTON Try-On Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        # CatVTON's expected resolution (portrait, not square!)
        self.width = 768
        self.height = 1024
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True
        )

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
                print("CatVTON loaded successfully.")
            except ImportError as e:
                print(f"Warning: CatVTON imports failed: {e}. Ensure CatVTON is cloned and in sys.path.")
            except Exception as e:
                print(f"Failed to load CatVTON: {e}")
        else:
            print("Warning: CUDA not found. CatVTON requires GPU. Running dummy fallback.")

    def run(self, person_image: Image.Image, garment_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Stage B: Real CatVTON Inference.
        person_image: Photo of the person
        garment_image: Product photo of the garment (white background, flat lay)
        mask: Binary mask of the region to replace on the person
        """
        if not self.pipeline:
            # Dummy fallback if model didn't load
            person_image = person_image.convert("RGBA")
            garment_image = garment_image.resize(person_image.size).convert("RGBA")
            mask = mask.resize(person_image.size).convert("L")
            result = Image.composite(garment_image, person_image, mask)
            return result.convert("RGB")

        orig_size = person_image.size  # save to restore at the end
        print("Running CatVTON inference...")
        generator = torch.Generator(device=self.device).manual_seed(42)

        # Blur the mask edges for smoother blending (as done in the official CatVTON app)
        mask_blurred = self.mask_processor.blur(mask, blur_factor=9)

        # CatVTON check_inputs:
        # - person + mask -> resize_and_crop to (width, height)
        # - garment       -> resize_and_padding to (width, height)
        # We pass PIL images directly and let CatVTON handle all resizing internally.
        result_image = self.pipeline(
            image=person_image.convert("RGB"),
            condition_image=garment_image.convert("RGB"),
            mask=mask_blurred,
            num_inference_steps=50,
            guidance_scale=2.5,
            height=self.height,
            width=self.width,
            generator=generator
        )[0]

        # Resize back to original person image size
        return result_image.resize(orig_size, Image.LANCZOS)
