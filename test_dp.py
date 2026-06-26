import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch

# Setup paths
repo_path = "/content/try-on-gpu/CatVTON" # If running in colab
# We are running this locally for testing syntax, so we mock it.
# Actually we can just write the logic.

def extend_mask_for_punjabi(base_mask: Image.Image, densepose_map: Image.Image) -> Image.Image:
    # Convert to numpy
    mask_np = np.array(base_mask)
    dp_np = np.array(densepose_map)
    
    # Resize dp_np to match mask_np if needed
    if dp_np.shape[:2] != mask_np.shape[:2]:
        dp_np = cv2.resize(dp_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Thigh labels in DensePose
    thigh_labels = [7, 8, 9, 10]
    
    # Create a boolean mask of thigh pixels
    thigh_mask = np.isin(dp_np, thigh_labels)
    
    # We want to keep only the upper ~70% of the thighs for a punjabi
    extension = np.zeros_like(mask_np)
    
    # Find bounding box of thighs
    y_idx, x_idx = np.where(thigh_mask)
    if len(y_idx) > 0:
        # Instead of global min/max, let's do it per column for a natural curve, 
        # or just globally for simplicity.
        # Let's do a simple vertical gradient cutoff.
        y_min = np.min(y_idx)
        y_max = np.max(y_idx)
        y_cutoff = y_min + int((y_max - y_min) * 0.70)
        
        # Apply cutoff
        valid_thighs = thigh_mask & (np.indices(dp_np.shape)[0] < y_cutoff)
        extension[valid_thighs] = 255
        
        # Morphological operations to smooth the extension
        kernel = np.ones((15, 15), np.uint8)
        extension = cv2.dilate(extension, kernel, iterations=1)
        
        # Add to base mask
        final_mask = np.clip(mask_np.astype(np.uint16) + extension.astype(np.uint16), 0, 255).astype(np.uint8)
        return Image.fromarray(final_mask)
        
    return base_mask

print("Syntax valid.")
