import requests
import base64
import time
import csv
import json
import os

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

def evaluate_via_api(url="https://pastime-confusion-job.ngrok-free.dev/"):
    # If run in colab, they might use ngrok or localhost
    # Let's support an env var or a default fallback
    url = os.environ.get("API_URL", url).rstrip('/')
    print(f"Connecting to API at {url}/evaluate...")
    
    # Using the same test images from test_infer.py (assume they exist in the root directory)
    # We navigate up one directory because this script is in gpu_worker/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    person_img_path = os.path.join(root_dir, "image.png")
    garment_img_path = os.path.join(root_dir, "pun.webp")
    
    if not os.path.exists(person_img_path) or not os.path.exists(garment_img_path):
        print(f"Test images not found! Please ensure '{person_img_path}' and '{garment_img_path}' exist.")
        return

    steps_to_test = [20, 30, 40, 50]
    scales_to_test = [1.5, 2.0, 2.5]
    
    results_file = os.path.join(root_dir, "experiment_results.tsv")
    
    # Initialize the TSV file with headers
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Steps', 'Guidance Scale', 'Inference Time (s)', 'Saved Image'])

    # Send individual requests for each combination to prevent ngrok timeouts (ERR_NGROK_3004)
    for steps in steps_to_test:
        for scale in scales_to_test:
            payload = {
                "person_image": encode_image(person_img_path),
                "garment_image": encode_image(garment_img_path),
                "garment_category": "upper",
                "garment_type": "shirt",
                "steps_list": [steps],
                "scales_list": [scale]
            }
            
            try:
                headers = {"ngrok-skip-browser-warning": "true"}
                print(f"Sending evaluation request for Steps={steps}, Scale={scale}...")
                response = requests.post(f"{url}/evaluate", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    continue
                
                item = results[0]
                time_taken = item["time_taken"]
                output_b64 = item["output_image"]
                
                # Save the image
                img_name = f"eval_s{steps}_g{scale}.jpg"
                img_path = os.path.join(root_dir, img_name)
                decode_and_save_image(output_b64, img_path)
                
                print(f"Success! Time: {time_taken:.2f}s | Saved: {img_name}")
                
                # Append to TSV file incrementally
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([steps, scale, f"{time_taken:.2f}", img_name])
                
            except Exception as e:
                print(f"Request failed for Steps={steps}, Scale={scale}: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"Response text: {response.text}")

    print(f"\nEvaluation complete! Results saved to {results_file}")

if __name__ == "__main__":
    evaluate_via_api()
