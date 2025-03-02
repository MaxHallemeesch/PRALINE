#!/bin/bash

python3 evaluateCPTAC.py \
        --src_path /project_antwerp/BRCA/ \
        --ref_file /project_antwerp/BRCA/cptac_brca_new.csv \
        --feature_path /project_antwerp/CPTAC-BRCA-resnet/ \
        --feature_dim 2048 \
        --save_dir output_v2 \
        --cohort TCGA-BRCA \
        --batch_size 16 \
        --k 5 \
        --log 0 \
        --save_on loss+corr \
        --stop_on loss+corr \
        --testing 0 \
        --model_type vis \
        --model_version Regular \
        --priorknowledge_type external \
        --gene_embeddings_path gene_embeddings \
        --gene_embeddings embeddings_BRCA_NMF \
        --lam 0 \
        --r_threshold 0.85 \
        --exp_name LateFusionBalanced/Resnet/NoPriorKnowledge\
        --checkpoint LateFusionBalanced/Resnet/NoPriorKnowledge\
        --load_splits 1 \