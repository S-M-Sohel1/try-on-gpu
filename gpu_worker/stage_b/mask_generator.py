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

    def generate_mask(self, person_image: Image.Image, category: str) -> Image.Image:
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
        if category == "upper":
            # Masking ONLY Upper-clothes (4) prevents the model from erasing the user's actual arms/hands.
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

        # Reduce dilation so the mask doesn't bleed into the hands or neck!
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return Image.fromarray(mask)
