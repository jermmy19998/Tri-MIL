from PIL import Image
import os

# 关闭 PIL 的最大像素限制
Image.MAX_IMAGE_PIXELS = None

# 输入图片路径列表
img_paths = [
    "/mnt/raid/zanzhuheng/working/sheep/heatmap_viz/5.ome_merged_pyramid/heatmap_phikon_v2_AB_MIL_OPA.png",
    "/mnt/raid/zanzhuheng/working/sheep/heatmap_viz/out_H2_merged_pyramid/heatmap_phikon_v2_AB_MIL_N-OPA.png",
]

# 输出目录（与原图同目录）
target_size = (1024, 1024)

for img_path in img_paths:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BICUBIC)

    save_path = img_path.replace(".png", "_1024.png")
    img.save(save_path)

    print(f"Saved: {save_path}")
