#!/bin/bash

python3 evaluateCPTAC.py \
        --src_path /project_antwerp/BRCA/ \
        --ref_file /project_antwerp/BRCA/cptac_brca_new.csv \
        --feature_path /project_antwerp/CPTAC-BRCA-uni/ \
        --feature_dim 1024 \
        --save_dir output_v2 \
        --cohort TCGA-BRCA \
        --batch_size 16 \
        --k 5 \
        --log 0 \
        --save_on loss+corr \
        --stop_on loss+corr \
        --testing 0 \
        --model_type vit \
        --model_version Balanced \
        --priorknowledge_type combined \
        --gene_embeddings_path gene_embeddings \
        --gene_embeddings embeddings_BRCA_NMF \
        --lam 0.9 \
        --r_threshold 0.85 \
        --exp_name Transformer/Uni/CombinedPriorKnowledge/lambda_0.9\
        --checkpoint Transformer/Uni/CombinedPriorKnowledge/lambda_0.9\
        --load_splits 1 \