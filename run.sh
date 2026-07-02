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


# Folder-first training with reference CSV
python tri_mil.py train \
  --yaml_path ./configs/AB_MIL.yaml \
  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
  --reference_csv ../tcga_brca_all_clean.csv \
  --slide_col slide_id \
  --label_col label \
  --dataset_name TCGA \
  --output_dir ./train_workspace \
  --k 3

# Folder-first training with label subfolders in raw WSI directory
# python tri_mil.py train \
#   --yaml_path ./configs/AB_MIL.yaml \
#   --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
#   --source_dir ../data \
#   --dataset_name TCGA \
#   --output_dir ./train_workspace \
#   --source_recursive \
#   --k 3

# Manual fallback: prepare CSVs and run train_mil.py directly if you need the old workflow

# Inference CSV from a plain feature folder
# Folder-first no-label inference
python tri_mil.py infer \
  --yaml_path ./configs/AB_MIL.yaml \
  --model_weight_path ./log_dir/TCGA/AB_MIL/time_2026-02-12-23-57_TCGA_AB_MIL_seed_2024/fold_1/Best_EPOCH_2.pth \
  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
  --test_log_dir ./test_nolabel \
  --no_label

# Folder-first heatmap generation
# If your WSIs are png/jpg, set --reader_type image and provide --mpp.
python tri_mil.py heatmap \
  --wsi_dir ../data \
  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
  --coord_dir ./trident_processed/20x_256px_0px_overlap/patches \
  --model_yaml ./configs/AB_MIL.yaml \
  --model_ckpt ./log_dir/TCGA/AB_MIL/time_2026-02-12-23-57_TCGA_AB_MIL_seed_2024/fold_1/Best_EPOCH_2.pth \
  --job_dir ./heatmap_viz \
  --blur
