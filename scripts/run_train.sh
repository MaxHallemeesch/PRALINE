#!/bin/bash

python3 hyperparameter_tuning.py \
        --src_path /project_antwerp/BRCA/ \
        --ref_file /project_antwerp/BRCA/brca.csv \
        --feature_path /project_antwerp/BRCA/features/ \
        --save_dir output \
        --cohort TCGA-BRCA \
        --exp_name BRCA_LateFusion_Unbalanced \
        --batch_size 16 \
        --k 5 \
        --train \
        --log 1 \
        --save_on loss+corr \
        --stop_on loss+corr \
        --testing 0


        # --checkpoint pretrained_models/model_best.pt\