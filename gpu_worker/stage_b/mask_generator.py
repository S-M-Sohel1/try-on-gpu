import os
import cv2
import numpy as np
import torch
from PIL import Image

# Garment types that are longer than a regular shirt and need downward mask extension
LONG_UPPER_GARMENTS = {"punjabi", "kurta", "kameez", "sherwani", "jubba", "thobe", "abaya"}

class MaskGenerator:
    def __init__(self, repo_path: str = None):
        """
        repo_path: Optional path to the locally downloaded zhengchong/CatVTON repo.
                   Used only if Segformer fails and AutoMasker fallback is needed.
        """
        self.segmenter = None
        self.automasker = None
        self.mask_processor = None
        self.repo_path = repo_path
        self.device = 0 if torch.cuda.is_available() else -1
        device_str = "cuda" if self.device == 0 else "cpu"

        # --- Primary: Segformer clothing segmenter ---
        try:
            from transformers import pipeline as hf_pipeline
            print("Initializing Segformer clothing segmenter (mattmdjaga/segformer_b2_clothes)...")
            self.segmenter = hf_pipeline(
                "image-segmentation",
                model="mattmdjaga/segformer_b2_clothes",
                device=self.device
            )
            print("Segformer initialized successfully.")
        except Exception as e:
            print(f"Warning: Segformer failed to load: {e}. Will try AutoMasker fallback.")

        # --- Secondary fallback: CatVTON AutoMasker (SCHP + DensePose) ---
        if not self.segmenter and repo_path and self.device == 0:
            try:
                from model.cloth_masker import AutoMasker
                from diffusers.image_processor import VaeImageProcessor

                densepose_ckpt = os.path.join(repo_path, "DensePose")
                schp_ckpt = os.path.join(repo_path, "SCHP")

                print("Initializing CatVTON AutoMasker as fallback (SCHP + DensePose)...")
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
                print("AutoMasker fallback initialized successfully.")
            except Exception as e:
                print(f"Warning: AutoMasker fallback also failed: {e}. Will use bounding-box masks.")
        elif not self.segmenter:
            print("Warning: No GPU or no repo_path. Using bounding-box mask fallback.")

    def _is_long_garment(self, garment_type: str) -> bool:
        return bool(garment_type and garment_type.strip().lower() in LONG_UPPER_GARMENTS)

    def _save_debug_mask(self, mask: Image.Image):
        try:
            mask.save("debug_mask.png")
            print("Debug mask saved to debug_mask.png")
        except Exception as e:
            print(f"Could not save debug mask: {e}")

    def _segformer_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """Generate mask using Segformer semantic clothing segmentation."""
        w, h = person_image.size
        is_long = self._is_long_garment(garment_type)

        results = self.segmenter(person_image)

        if category == "upper":
            if is_long:
                # For long garments, include the shirt body + arms for a wider base
                target_labels = {"Upper-clothes", "Left-arm", "Right-arm"}
            else:
                # For regular shirts/t-shirts, mask only the garment body (not arms)
                target_labels = {"Upper-clothes"}
        elif category == "lower":
            target_labels = {"Pants", "Skirt", "Left-leg", "Right-leg"}
        else:  # overall / dress
            target_labels = {"Upper-clothes", "Pants", "Dress", "Skirt", "Left-arm", "Right-arm"}

        mask = np.zeros((h, w), dtype=np.uint8)
        for result in results:
            if result["label"] in target_labels:
                label_mask = np.array(result["mask"])
                mask[label_mask > 0] = 255

        # For long garments, extend the mask downward from the shirt hem to cover thighs
        if is_long and category == "upper":
            y_shirt = np.where(mask > 0)[0]
            if len(y_shirt) > 0:
                shirt_bottom = int(np.max(y_shirt))
                extension_bottom = int(h * 0.75)
                if extension_bottom > shirt_bottom:
                    x_at_bottom = np.where(mask[shirt_bottom, :] > 0)[0]
                    if len(x_at_bottom) > 0:
                        x_left = max(0, int(np.min(x_at_bottom)) - 10)
                        x_right = min(w, int(np.max(x_at_bottom)) + 10)
                        mask[shirt_bottom:extension_bottom, x_left:x_right] = 255

        # Dilate to smooth edges — larger kernel for long garments
        kernel_size = 12 if is_long else 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return Image.fromarray(mask)

    def _automasker_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """Generate mask using CatVTON's AutoMasker (secondary fallback)."""
        # Use 'upper' base type even for long garments to avoid over-masking
        base_mask_type = "upper" if category == "upper" else category
        result = self.automasker(person_image, mask_type=base_mask_type)
        mask = result["mask"]  # PIL Image
        print(f"AutoMasker generated mask with type='{base_mask_type}'")
        return mask

    def _bbox_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """Last-resort bounding-box mask."""
        w, h = person_image.size
        is_long = self._is_long_garment(garment_type)
        mask = np.zeros((h, w), dtype=np.uint8)
        if is_long and category == "upper":
            mask[int(h * 0.10):int(h * 0.78), int(w * 0.10):int(w * 0.90)] = 255
        elif category == "upper":
            mask[int(h * 0.15):int(h * 0.65), int(w * 0.15):int(w * 0.85)] = 255
        elif category == "lower":
            mask[int(h * 0.50):int(h * 0.95), int(w * 0.15):int(w * 0.85)] = 255
        else:
            mask[int(h * 0.15):int(h * 0.95), int(w * 0.15):int(w * 0.85)] = 255
        return Image.fromarray(mask)

    def generate_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """
        Generates a cloth-agnostic mask for the region the garment will be applied to.

        Priority:
          1. Segformer (mattmdjaga/segformer_b2_clothes) — precise per-label mask
          2. CatVTON AutoMasker (SCHP + DensePose) — if Segformer unavailable
          3. Bounding-box fallback — if both models unavailable
        """
        is_long = self._is_long_garment(garment_type)
        print(f"Generating mask | category='{category}' | garment_type='{garment_type}' | long_garment={is_long}")

        mask = None

        # 1. Segformer (primary)
        if self.segmenter:
            try:
                mask = self._segformer_mask(person_image, category, garment_type)
                print("Mask generated via Segformer (primary).")
            except Exception as e:
                print(f"Segformer mask failed: {e}. Trying AutoMasker fallback.")

        # 2. AutoMasker (secondary fallback)
        if mask is None and self.automasker:
            try:
                mask = self._automasker_mask(person_image, category, garment_type)
                print("Mask generated via AutoMasker (fallback).")
            except Exception as e:
                print(f"AutoMasker mask failed: {e}. Falling back to bounding-box.")

        # 3. Bounding-box (last resort)
        if mask is None:
            mask = self._bbox_mask(person_image, category, garment_type)
            print("Mask generated via bounding-box (last resort).")

        # Save debug mask for inspection
        self._save_debug_mask(mask)

        return mask
