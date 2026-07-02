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


# Case 1:
# Flat feature directory + reference CSV
python prepare_dataset_csv.py \
  --mode train_flat \
  --reference_csv ../tcga_brca_all_clean.csv \
  --slide_col slide_id \
  --label_col label \
  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
  --output_csv ./train_base.csv

# Case 2:
# Raw WSI directory contains label subfolders, features are stored flat after extraction
# python prepare_dataset_csv.py \
#   --mode train_label_dirs \
#   --source_dir ../data \
#   --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
#   --output_csv ./train_base.csv \
#   --source_recursive

# Case 3:
# Feature directory itself already contains label subfolders
# python prepare_dataset_csv.py \
#   --mode train_label_dirs \
#   --feature_dir ./labeled_features \
#   --output_csv ./train_base.csv

# Build k-fold training CSVs from slide_path,label
python ./split_scripts/split_datasets_k_fold_train_val.py \
  --csv_path ./train_base.csv \
  --dataset_name TCGA \
  --save_dir ./datasets

# Train
python train_mil.py --yaml_path ./configs/AB_MIL.yaml

# Inference CSV from a plain feature folder
python prepare_dataset_csv.py \
  --mode infer \
  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_vit \
  --output_csv ./test.csv

# No-label inference
python infer_mil.py \
  --yaml_path ./configs/AB_MIL.yaml \
  --test_dataset_csv ./test.csv \
  --model_weight_path ./log_dir/TCGA/AB_MIL/time_2026-02-12-23-57_TCGA_AB_MIL_seed_2024/fold_1/Best_EPOCH_2.pth \
  --test_log_dir ./test_nolabel \
  --no_label

# Heatmap
python ./draw_heatmap/draw_heatmap.py --config ./draw_heatmap/heatmap.yaml
