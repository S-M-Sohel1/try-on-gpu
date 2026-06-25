import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

class TextureMapper:
    def __init__(self):
        print("Initializing Stage A Generative Texture Mapper (SD Img2Img)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SD 1.5 for image-to-image texture baking
        model_id = "runwayml/stable-diffusion-v1-5"
        if self.device == "cuda":
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None
            )
            # Offload to CPU when not in use to save VRAM for CatVTON and Segformer
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = None # For local testing without GPU, we fallback to OpenCV only
            print("Warning: CUDA not found. Stage A Generative baking disabled. Using OpenCV fallback.")

    def apply_fabric(self, fabric_image: Image.Image, category: str, template_id: str = None) -> Image.Image:
        """
        Stage A: Texture Mapper.
        Tiles the fabric using OpenCV, then uses Stable Diffusion Img2Img 
        to bake realistic lighting, folds, and texture onto the garment.
        """
        # Step 1: OpenCV base tiling
        fabric_cv = cv2.cvtColor(np.array(fabric_image), cv2.COLOR_RGB2BGR)
        target_size = (768, 768) # Larger resolution for better CatVTON quality
        template_mask = np.zeros((*target_size, 3), dtype=np.uint8)
        
        if category == "upper":
            # Simple shirt silhouette scaled for 768x768
            pts = np.array([[225, 150], [543, 150], [675, 300], [600, 450], 
                            [525, 375], [525, 750], [243, 750], [243, 375], 
                            [168, 450], [93, 300]], np.int32)
            cv2.fillPoly(template_mask, [pts], (255, 255, 255))
        elif category == "lower":
            # Simple pant silhouette scaled for 768x768
            pts = np.array([[225, 150], [543, 150], [600, 750], [420, 750], 
                            [384, 450], [348, 750], [168, 750]], np.int32)
            cv2.fillPoly(template_mask, [pts], (255, 255, 255))
        else:
            cv2.rectangle(template_mask, (150, 75), (618, 750), (255, 255, 255), -1)

        h, w = fabric_cv.shape[:2]
        repeats_y = (target_size[0] // h) + 1
        repeats_x = (target_size[1] // w) + 1
        tiled_fabric = np.tile(fabric_cv, (repeats_y, repeats_x, 1))
        tiled_fabric = tiled_fabric[:target_size[0], :target_size[1]]

        mask_gray = cv2.cvtColor(template_mask, cv2.COLOR_BGR2GRAY)
        result_cv = cv2.bitwise_and(tiled_fabric, tiled_fabric, mask=mask_gray)
        
        bg = np.ones_like(result_cv, dtype=np.uint8) * 255
        inv_mask = cv2.bitwise_not(mask_gray)
        bg = cv2.bitwise_and(bg, bg, mask=inv_mask)
        init_image_cv = cv2.add(result_cv, bg)
        init_image_pil = Image.fromarray(cv2.cvtColor(init_image_cv, cv2.COLOR_BGR2RGB))

        # Step 2: Generative Texture Baking (Img2Img)
        if self.pipe:
            garment_name = "shirt" if category == "upper" else "pants"
            prompt = f"a high quality product photography of a flat lay {garment_name} garment, realistic fabric folds, shadows, highly detailed"
            negative_prompt = "person, body, head, text, watermark, bad quality, 3d render"
            
            # Strength 0.35 keeps the pattern intact but adds realistic folds and lighting
            generated = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image_pil,
                strength=0.35,
                guidance_scale=7.5
            ).images[0]
            
            # Re-apply the mask to ensure crisp white background
            gen_cv = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            gen_masked = cv2.bitwise_and(gen_cv, gen_cv, mask=mask_gray)
            gen_final = cv2.add(gen_masked, bg)
            return Image.fromarray(cv2.cvtColor(gen_final, cv2.COLOR_BGR2RGB))
        
        return init_image_pil
