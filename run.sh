# Mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT

# Preprocess slides and extract features
# If your WSIs are stored under nested label folders, add: --search_nested
nohup python run_batch_of_slides.py \
  --task all \
  --wsi_dir ../data \
  --job_dir ./trident_processed \
  --patch_encoder vit \
  --mag 20 \
  --patch_size 256 \
  > ./processed.log 2>&1 &


# Unified YAML workflow
python train_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python test_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
python infer_mil.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml --no_label
python ./draw_heatmap/draw_heatmap.py --yaml_path ./configs/TRI_MIL_PIPELINE.yaml
