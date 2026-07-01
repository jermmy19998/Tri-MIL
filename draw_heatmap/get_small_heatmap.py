from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None 
# Input and output paths
img_paths = [
    "/mnt/raid/zanzhuheng/working/sheep/heatmap_viz/out_H2_merged_pyramid/heatmap_phikon_v2_CLAM_MB_MIL_N-OPA.png",
]

output_paths = [
    "/mnt/raid/zanzhuheng/working/sheep/heatmap_viz/out_H2_merged_pyramid/heatmap_phikon_v2_CLAM_MB_MIL_N-OPA1024.png"
]

target_size = (1024, 1024)

for inp, outp in zip(img_paths, output_paths):
    # Open image
    img = Image.open(inp)
    # Resize to 1024x1024 using Lanczos for better quality
    img_resized = img.resize(target_size, Image.LANCZOS)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    # Save resized image
    img_resized.save(outp)
    print(f"Saved resized image to {outp}")
