import torch
from diffusers import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

class CatVTONRunner:
    def __init__(self):
        print("Initializing Real CatVTON Try-On Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.repo_path = None  # exposed so MaskGenerator can locate SCHP/DensePose weights
        # CatVTON's native training resolution (portrait)
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
                # CatVTON repo must be cloned and added to sys.path in the Colab notebook
                from model.pipeline import CatVTONPipeline

                print("Downloading CatVTON weights from HuggingFace... (cached after first run)")
                self.repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
                print(f"CatVTON weights at: {self.repo_path}")

                self.pipeline = CatVTONPipeline(
                    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                    attn_ckpt=self.repo_path,
                    attn_ckpt_version="mix",
                    weight_dtype=torch.float16,
                    use_tf32=True,
                    skip_safety_check=True,
                    device=self.device
                )
                print("CatVTON loaded successfully.")
            except ImportError as e:
                print(f"Warning: CatVTON imports failed: {e}. Ensure CatVTON repo is cloned and in sys.path.")
            except Exception as e:
                print(f"Failed to load CatVTON: {e}")
        else:
            print("Warning: CUDA not found. CatVTON requires GPU. Running dummy fallback.")

    def run(self, person_image: Image.Image, garment_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Runs CatVTON inpainting.
        - person_image: The photo of the person.
        - garment_image: Flat-lay / white-background product photo of the garment.
        - mask: Binary mask of the region to replace (from AutoMasker).
        """
        if not self.pipeline:
            # Dummy passthrough fallback (no GPU / failed load)
            person_image = person_image.convert("RGBA")
            garment_image = garment_image.resize(person_image.size).convert("RGBA")
            mask_l = mask.resize(person_image.size).convert("L")
            result = Image.composite(garment_image, person_image, mask_l)
            return result.convert("RGB")

        orig_size = person_image.size
        print("Running CatVTON inference...")
        generator = torch.Generator(device=self.device).manual_seed(42)

        # Blur mask edges for smooth blending — CatVTON's recommended value is 9
        mask_blurred = self.mask_processor.blur(mask, blur_factor=9)

        # CatVTON internally handles resizing:
        #   person + mask → resize_and_crop  to (width, height)
        #   garment       → resize_and_padding to (width, height)
        result_image = self.pipeline(
            image=person_image.convert("RGB"),
            condition_image=garment_image.convert("RGB"),
            mask=mask_blurred,
            num_inference_steps=50,
            guidance_scale=2.5,   # CatVTON's own default — do not increase
            height=self.height,
            width=self.width,
            generator=generator
        )[0]

        return result_image.resize(orig_size, Image.LANCZOS)
