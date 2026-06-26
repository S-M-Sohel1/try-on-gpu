import os
import cv2
import numpy as np
import torch
from PIL import Image

# Garment types that require a longer / full-body mask
LONG_UPPER_GARMENTS = {"punjabi", "kurta", "kameez", "sherwani", "jubba", "thobe", "abaya"}

class MaskGenerator:
    def __init__(self, repo_path: str = None):
        """
        repo_path: Path to the locally downloaded zhengchong/CatVTON repo 
                   (returned by snapshot_download). Contains SCHP/ and DensePose/ folders.
        """
        self.automasker = None
        self.mask_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_str = "cuda" if self.device == "cuda" else "cpu"

        if repo_path and self.device == "cuda":
            try:
                # AutoMasker lives inside the CatVTON repo — must be in sys.path
                from model.cloth_masker import AutoMasker
                from diffusers.image_processor import VaeImageProcessor

                densepose_ckpt = os.path.join(repo_path, "DensePose")
                schp_ckpt = os.path.join(repo_path, "SCHP")

                print("Initializing CatVTON AutoMasker (SCHP + DensePose)...")
                self.automasker = AutoMasker(
                    densepose_ckpt=densepose_ckpt,
                    schp_ckpt=schp_ckpt,
                    device=device_str
                )
                self.mask_processor = VaeImageProcessor(
                    vae_scale_factor=8,
                    do_normalize=False,
                    do_binarize=True,
                    do_convert_grayscale=True
                )
                print("AutoMasker initialized successfully.")
            except ImportError as e:
                print(f"Warning: AutoMasker import failed: {e}. Falling back to bounding-box mask.")
            except Exception as e:
                print(f"Warning: AutoMasker init failed: {e}. Falling back to bounding-box mask.")
        else:
            if self.device != "cuda":
                print("Warning: No GPU detected. AutoMasker disabled — using bounding-box fallback.")
            else:
                print("Warning: No repo_path given to MaskGenerator. Using bounding-box fallback.")

    def _garment_to_mask_type(self, category: str, garment_type: str = None) -> str:
        """
        Maps our garment_category + garment_type to CatVTON's mask_type.
        CatVTON mask_type options: 'upper', 'lower', 'overall', 'inner', 'outer'
        """
        if category == "lower":
            return "lower"
        if category == "overall":
            return "overall"
        # category == "upper"
        if garment_type and garment_type.strip().lower() in LONG_UPPER_GARMENTS:
            # Long upper garments need the 'overall' mask so CatVTON can paint
            # the garment body all the way down past the waist
            return "overall"
        return "upper"

    def generate_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """
        Generates a cloth-agnostic mask for the region the garment will be applied to.
        Uses CatVTON's AutoMasker (SCHP + DensePose) when available.
        Falls back to a simple bounding-box mask if AutoMasker is unavailable.
        """
        w, h = person_image.size
        mask_type = self._garment_to_mask_type(category, garment_type)
        print(f"Generating mask with mask_type='{mask_type}' (garment_type='{garment_type}')")

        # --- CatVTON AutoMasker path (preferred) ---
        if self.automasker:
            try:
                result = self.automasker(person_image, mask_type=mask_type)
                mask = result["mask"]  # PIL Image, already properly computed
                return mask
            except Exception as e:
                print(f"AutoMasker inference failed: {e}. Falling back to bounding-box.")

        # --- Bounding-box fallback ---
        print("Using bounding-box fallback mask.")
        mask = np.zeros((h, w), dtype=np.uint8)
        if mask_type == "upper":
            mask[int(h*0.10):int(h*0.58), int(w*0.10):int(w*0.90)] = 255
        elif mask_type == "overall":
            mask[int(h*0.10):int(h*0.80), int(w*0.10):int(w*0.90)] = 255
        elif mask_type == "lower":
            mask[int(h*0.45):int(h*0.95), int(w*0.10):int(w*0.90)] = 255
        else:
            mask[int(h*0.10):int(h*0.95), int(w*0.10):int(w*0.90)] = 255
        return Image.fromarray(mask)
