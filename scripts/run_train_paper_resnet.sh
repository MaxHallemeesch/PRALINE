#!/bin/bash

python3 main_paper.py \
        --src_path /project_antwerp/BRCA/ \
        --ref_file /project_antwerp/BRCA/brca.csv \
        --feature_path /project_antwerp/BRCA/features/ \
        --feature_dim 2048 \
        --save_dir output_v2 \
        --cohort TCGA-BRCA \
        --batch_size 16 \
        --k 5 \
        --log 1 \
        --save_on loss+corr \
        --stop_on loss+corr \
        --testing 0 \
        --train \
        --model_type vis \
        --exp_name LateFusionBalanced/Resnet/ExternalPriorKnowledge/lambda_0.9 \
        --model_version Balanced \
        --lam 0.9 \
        --priorknowledge_type external \
        --gene_embeddings_path gene_embeddings \
        --gene_embeddings embeddings_BRCA_NMF \
        --r_threshold 0.85 \
        #--gtex 1 \
        #--checkpoint /project_antwerp/sequoia_models/gtex_model.pt
        #--r_threshold 0.85 \
        #--exp_name LateFusionBalanced/InternalPriorKnowledge/R_0.65/NMF/lambda_0.5 \
        #--gtex 0 \
        #--checkpoint pretrained_models/model_best.pt\