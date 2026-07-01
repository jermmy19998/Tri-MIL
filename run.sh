# 镜像设置
export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT

# 切割patch并提取特征
nohup python run_batch_of_slides.py \
  --task all \
  --wsi_dir ../data \
  --job_dir ./trident_processed \
  --patch_encoder resnet50 \
  --mag 20 \
  --patch_size 256 \
  > ./processed.log 2>&1 &


# 生成训练所需的csv
python gen_train_csv.py --reference_csv ../tcga_brca_all_clean.csv  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_resnet50 --output_csv ./train.csv

# 生成交叉验证csv(只训练并验证不test)
python ./split_scripts/split_datasets_k_fold_train_val.py --csv_path ./train.csv --dataset_name TCGA --save_dir ./datasets

# 开始训练
python train_mil.py --yaml_path ./configs/AB_MIL.yaml


# 生成测试所需的csv(注意一般测试数据为不同的一批模型没见过的数据，这里直接用训练的作为演示)
python gen_test_csv.py --reference_csv ../tcga_brca_all_clean.csv  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_resnet50 --output_csv ./test.csv

# 无标签推理模式
python gen_test_csv.py --reference_csv ../tcga_brca_all_clean.csv  --feature_dir ./trident_processed/20x_256px_0px_overlap/features_resnet50 --output_csv ./test_nolabel.csv --no_label

python infer_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv ./test.csv --model_weight_path ./log_dir/TCGA/AB_MIL/time_2026-02-12-23-57_TCGA_AB_MIL_seed_2024/fold_1/Best_EPOCH_2.pth --test_log_dir ./test



python infer_mil.py --yaml_path ./configs/AB_MIL.yaml --test_dataset_csv ./test_nolabel.csv --model_weight_path ./log_dir/TCGA/AB_MIL/time_2026-02-12-23-57_TCGA_AB_MIL_seed_2024/fold_1/Best_EPOCH_2.pth --test_log_dir ./test_nolabel --no_label

python ./draw_heatmap/draw_heatmap.py --config ./draw_heatmap/heatmap.yaml