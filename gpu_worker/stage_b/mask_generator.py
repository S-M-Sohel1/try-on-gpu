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
            self.segmenter = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=self.device)
        except Exception as e:
            print(f"Warning: Failed to load segformer: {e}. Masking will fallback.")
            self.segmenter = None

        # Segformer labels: Background(0), Hat(1), Hair(2), Sunglasses(3), Upper-clothes(4), 
        # Skirt(5), Pants(6), Dress(7), Belt(8), Left-shoe(9), Right-shoe(10), Face(11), 
        # Left-leg(12), Right-leg(13), Left-arm(14), Right-arm(15), Left-hand(16), Right-hand(17)

    def generate_mask(self, person_image: Image.Image, category: str, garment_type: str = None) -> Image.Image:
        """
        Stage B: Auto-Mask Generation.
        Generates a binary mask of the clothing region using Segformer.
        """
        if not self.segmenter:
            # Fallback to dummy
            w, h = person_image.size
            mask = np.zeros((h, w), dtype=np.uint8)
            if category == "upper":
                mask[int(h*0.2):int(h*0.6), int(w*0.25):int(w*0.75)] = 255
            elif category == "lower":
                mask[int(h*0.5):int(h*0.9), int(w*0.25):int(w*0.75)] = 255
            return Image.fromarray(mask)

        # Run segmentation
        results = self.segmenter(person_image)
        
        w, h = person_image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine labels based on category
        target_labels = []
        is_punjabi = category == "upper" and garment_type and "punjabi" in garment_type.lower()
        
        if category == "upper":
            if is_punjabi:
                # For Punjabi, we MUST mask the arms so long sleeves can be drawn, 
                # and we need to expand the mask downwards to cover the upper legs.
                target_labels = ["Upper-clothes", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Pants"]
            else:
                # Standard shirt: mask only the existing shirt to preserve original arms
                target_labels = ["Upper-clothes"]
        elif category == "lower":
            # 6: Pants, 5: Skirt, 12: Left-leg, 13: Right-leg
            target_labels = ["Pants", "Skirt", "Left-leg", "Right-leg"]
        else:
            target_labels = ["Upper-clothes", "Pants", "Dress", "Skirt"]

        for result in results:
            if result['label'] in target_labels:
                # result['mask'] is a PIL image
                label_mask = np.array(result['mask'])
                mask[label_mask > 0] = 255

        if is_punjabi:
            # If Punjabi, we masked pants/legs to allow the long body to be drawn,
            # but we ONLY want the top half of the pants/legs to be masked!
            # Let's find the bounding box of the upper-clothes and cut off the mask below the knees.
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                top_y = np.min(y_indices)
                bottom_y = np.max(y_indices)
                # Cut off the bottom 30% of the masked region so the lower legs/shoes are preserved
                cutoff_y = int(top_y + (bottom_y - top_y) * 0.7)
                mask[cutoff_y:, :] = 0

            # Slightly larger dilation for Punjabi to blend the new long edges
            kernel = np.ones((10, 10), np.uint8)
        else:
            # Standard minimal dilation
            kernel = np.ones((5, 5), np.uint8)
            
        mask = cv2.dilate(mask, kernel, iterations=1)

        return Image.fromarray(mask)
