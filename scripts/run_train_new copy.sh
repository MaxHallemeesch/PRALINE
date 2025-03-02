#!/bin/bash

python3 main_new.py \
        --src_path /project_antwerp/BRCA/ \
        --ref_file /project_antwerp/BRCA/brca.csv \
        --feature_path /project_antwerp/BRCA-ctrans \
        --feature_dim 768 \
        --save_dir output_v2 \
        --cohort TCGA-BRCA \
        --batch_size 16 \
        --k 5 \
        --log 0 \
        --save_on loss+corr \
        --stop_on loss+corr \
        --testing 0 \
        --log 1 \
        --train \
        --model_type vis \
        --model_version Balanced \
        --priorknowledge_type internal \
        --gene_embeddings_path gene_embeddings \
        --gene_embeddings embeddings_BRCA_NMF \
        --r_threshold 0.65 \
        --lam 0.5 \
        --exp_name LateFusionBalanced/InternalPriorKnowledge/R_0.65/NMF/lambda_0.5 \
