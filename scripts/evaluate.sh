#!/bin/bash

python3 main_paper.py \
        --exp_name /exp_name \
        --feature_path /feature_path \
        --feature_dim 1024 \
        --ref_file /groundtruth \
        --save_dir /save_dir \
        --cohort TCGA-BRCA \
        --batch_size 16 \
        --k 5 \
        --model_type vis \
        --model_version PK \
        --priorknowledge_type internal \
        --gene_embeddings_path /gene_embeddings_path \
        --lam 0.1 \
        --checkpoint /path_to_stored_model \
        --load_splits 1 \