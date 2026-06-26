import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

class MaskGenerator:
    def __init__(self):
        print("Initializing AI Mask Generator (Segformer)...")
        self.device = 0 if torch.cuda.is_available() else -1

        try:
            self.segmenter = pipeline(
                "image-segmentation",
                model="mattmdjaga/segformer_b2_clothes",
                device=self.device
            )
        except Exception as e:
            print(f"Warning: Failed to load segformer: {e}. Masking will fallback.")
            self.segmenter = None

        # Segformer labels:
        # Background(0), Hat(1), Hair(2), Sunglasses(3), Upper-clothes(4),
        # Skirt(5), Pants(6), Dress(7), Belt(8), Left-shoe(9), Right-shoe(10), Face(11),
        # Left-leg(12), Right-leg(13), Left-arm(14), Right-arm(15), Left-hand(16), Right-hand(17)

    def generate_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """
        Stage B: Auto-Mask Generation.
        Generates a binary mask of the region on the person where the garment will be placed.
        """
        w, h = person_image.size

        if not self.segmenter:
            # Fallback to bounding box
            mask = np.zeros((h, w), dtype=np.uint8)
            if category == "upper":
                mask[int(h*0.15):int(h*0.65), int(w*0.15):int(w*0.85)] = 255
            elif category == "lower":
                mask[int(h*0.5):int(h*0.95), int(w*0.15):int(w*0.85)] = 255
            else:
                mask[int(h*0.15):int(h*0.95), int(w*0.15):int(w*0.85)] = 255
            return Image.fromarray(mask)

        # Run segmentation
        results = self.segmenter(person_image)

        mask = np.zeros((h, w), dtype=np.uint8)
        is_punjabi = bool(garment_type and "punjabi" in garment_type.lower())
        is_kurta = bool(garment_type and "kurta" in garment_type.lower())
        is_long_garment = is_punjabi or is_kurta

        if category == "upper":
            if is_long_garment:
                # For long garments (punjabi/kurta), mask shirt + both arms + upper half of lower body
                target_labels = ["Upper-clothes", "Left-arm", "Right-arm"]
            else:
                # Standard: only mask the existing shirt. Arms stay intact.
                target_labels = ["Upper-clothes"]
        elif category == "lower":
            target_labels = ["Pants", "Skirt", "Left-leg", "Right-leg"]
        else:  # overall / dress
            target_labels = ["Upper-clothes", "Pants", "Dress", "Skirt", "Left-arm", "Right-arm"]

        for result in results:
            if result['label'] in target_labels:
                label_mask = np.array(result['mask'])
                mask[label_mask > 0] = 255

        if is_long_garment:
            # Extend the mask downward to cover the upper legs so the long body can be painted
            # Find bottom edge of the current shirt mask
            y_shirt = np.where(mask > 0)[0]
            if len(y_shirt) > 0:
                shirt_bottom = int(np.max(y_shirt))
                # Extend the mask to 75% down the full image height
                extension_bottom = int(h * 0.75)
                if extension_bottom > shirt_bottom:
                    # Use the horizontal extent of the shirt at its bottom to define the extension
                    x_at_bottom = np.where(mask[shirt_bottom, :] > 0)[0]
                    if len(x_at_bottom) > 0:
                        x_left = max(0, int(np.min(x_at_bottom)) - 10)
                        x_right = min(w, int(np.max(x_at_bottom)) + 10)
                        mask[shirt_bottom:extension_bottom, x_left:x_right] = 255

            kernel = np.ones((12, 12), np.uint8)
        else:
            kernel = np.ones((5, 5), np.uint8)

        mask = cv2.dilate(mask, kernel, iterations=1)
        return Image.fromarray(mask)
