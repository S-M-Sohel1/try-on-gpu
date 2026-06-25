import requests
import base64
import json
import time

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        b64_str = base64.b64encode(img_data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

def decode_and_save_image(b64_str, output_path):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    image_data = base64.b64decode(b64_str)
    with open(output_path, "wb") as f:
        f.write(image_data)

def test_inference(url="http://127.0.0.1:8000"):
    # Create dummy images for testing
    from PIL import Image
    import numpy as np

    person_img = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 200)
    person_img.save("dummy_person.jpg")
    fabric_img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 100)
    fabric_img.save("dummy_fabric.jpg")

    payload = {
        "person_image": encode_image("dummy_person.jpg"),
        "fabric_image": encode_image("dummy_fabric.jpg"),
        "garment_category": "upper"
    }

    print(f"Sending request to {url}/infer...")
    start = time.time()
    try:
        response = requests.post(f"{url}/infer", json=payload)
        response.raise_for_status()
        data = response.json()
        
        output_base64 = data.get("output_image")
        decode_and_save_image(output_base64, "output_result.jpg")
        print(f"Success! Output saved to output_result.jpg in {time.time() - start:.2f} seconds.")
    except Exception as e:
        print(f"Request failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response text: {response.text}")

if __name__ == "__main__":
    test_inference()
