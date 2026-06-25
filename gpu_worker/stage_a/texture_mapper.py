import cv2
import numpy as np
from PIL import Image

class TextureMapper:
    def __init__(self):
        # We could load generic templates here if needed.
        pass

    def apply_fabric(self, fabric_image: Image.Image, category: str, template_id: str = None) -> Image.Image:
        """
        Stage A: Texture Mapper (Option B fallback).
        Takes a fabric image and maps it onto a flat garment template.
        For MVP, we do a basic tile and crop/mask.
        """
        # Convert PIL to cv2
        fabric_cv = cv2.cvtColor(np.array(fabric_image), cv2.COLOR_RGB2BGR)
        
        # In a real scenario, we load a template mask based on category and template_id.
        # Here we will generate a synthetic simple silhouette mask for demonstration.
        target_size = (512, 512)
        template_mask = np.zeros((*target_size, 3), dtype=np.uint8)
        
        if category == "upper":
            # Draw a simple shirt silhouette
            pts = np.array([[150, 100], [362, 100], [450, 200], [400, 300], 
                            [350, 250], [350, 500], [162, 500], [162, 250], 
                            [112, 300], [62, 200]], np.int32)
            cv2.fillPoly(template_mask, [pts], (255, 255, 255))
        elif category == "lower":
            # Draw simple pant silhouette
            pts = np.array([[150, 100], [362, 100], [400, 500], [280, 500], 
                            [256, 300], [232, 500], [112, 500]], np.int32)
            cv2.fillPoly(template_mask, [pts], (255, 255, 255))
        else:
            # Overall or generic
            cv2.rectangle(template_mask, (100, 50), (412, 500), (255, 255, 255), -1)

        # Tile the fabric to cover the target size
        h, w = fabric_cv.shape[:2]
        repeats_y = (target_size[0] // h) + 1
        repeats_x = (target_size[1] // w) + 1
        tiled_fabric = np.tile(fabric_cv, (repeats_y, repeats_x, 1))
        tiled_fabric = tiled_fabric[:target_size[0], :target_size[1]]

        # Apply the mask
        mask_gray = cv2.cvtColor(template_mask, cv2.COLOR_BGR2GRAY)
        result_cv = cv2.bitwise_and(tiled_fabric, tiled_fabric, mask=mask_gray)
        
        # Background could be white instead of black
        bg = np.ones_like(result_cv, dtype=np.uint8) * 255
        inv_mask = cv2.bitwise_not(mask_gray)
        bg = cv2.bitwise_and(bg, bg, mask=inv_mask)
        result_cv = cv2.add(result_cv, bg)

        # Convert back to PIL
        result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        return result_pil
