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
        Uses CatVTON's AutoMasker (SCHP + DensePose).
        For long garments (Punjabi), we start with 'upper' mask and intelligently extend
        it over the thighs using DensePose mapping.
        """
        w, h = person_image.size
        
        # We always start with "upper" or "lower" to ensure AutoMasker protects the correct half
        base_mask_type = "upper" if category == "upper" else category
        is_long_garment = (category == "upper" and garment_type and garment_type.strip().lower() in LONG_UPPER_GARMENTS)
        
        print(f"Generating mask with base_mask_type='{base_mask_type}' | long_garment={is_long_garment}")

        # --- CatVTON AutoMasker path (preferred) ---
        if self.automasker:
            try:
                result = self.automasker(person_image, mask_type=base_mask_type)
                mask = result["mask"]  # PIL Image
                
                # Smart Extension for Punjabi/Kurta
                if is_long_garment:
                    print("Applying DensePose mask extension for long garment (Punjabi/Kurta)...")
                    mask_np = np.array(mask)
                    dp_np = np.array(result["densepose"])
                    
                    if dp_np.shape[:2] != mask_np.shape[:2]:
                        dp_np = cv2.resize(dp_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # DensePose Labels: Thighs (7,8,9,10), Hands (3,4)
                    thigh_mask = np.isin(dp_np, [7, 8, 9, 10])
                    hands_mask = np.isin(dp_np, [3, 4])
                    
                    y_idx, _ = np.where(thigh_mask)
                    if len(y_idx) > 0:
                        y_min, y_max = np.min(y_idx), np.max(y_idx)
                        # Extend down 70% of the thighs
                        y_cutoff = y_min + int((y_max - y_min) * 0.70)
                        
                        valid_thighs = thigh_mask & (np.indices(dp_np.shape)[0] < y_cutoff)
                        
                        extension = np.zeros_like(mask_np)
                        extension[valid_thighs] = 255
                        
                        # Remove hands from extension (strict protection)
                        extension[hands_mask] = 0
                        
                        # Smooth the extension slightly
                        kernel = np.ones((11, 11), np.uint8)
                        extension = cv2.dilate(extension, kernel, iterations=1)
                        
                        # Combine with base mask
                        mask_np = np.clip(mask_np.astype(np.uint16) + extension.astype(np.uint16), 0, 255).astype(np.uint8)
                        mask = Image.fromarray(mask_np)
                
                return mask
            except Exception as e:
                print(f"AutoMasker inference failed: {e}. Falling back to bounding-box.")

        # --- Bounding-box fallback ---
        print("Using bounding-box fallback mask.")
        mask = np.zeros((h, w), dtype=np.uint8)
        if is_long_garment:
            mask[int(h*0.10):int(h*0.80), int(w*0.10):int(w*0.90)] = 255
        elif category == "upper":
            mask[int(h*0.10):int(h*0.58), int(w*0.10):int(w*0.90)] = 255
        elif category == "lower":
            mask[int(h*0.45):int(h*0.95), int(w*0.10):int(w*0.90)] = 255
        else:
            mask[int(h*0.10):int(h*0.95), int(w*0.10):int(w*0.90)] = 255
        return Image.fromarray(mask)
