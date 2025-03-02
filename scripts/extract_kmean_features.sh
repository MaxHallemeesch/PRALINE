#!/bin/bash
python3 kmean_features.py \
        --ref_file /project_antwerp/BRCA/brca.csv  \
        --patch_data_path /project_antwerp/BRCA/patches/ \
        --feature_path /project_antwerp/BRCA/features/  \
        --num_clusters 100 \
        --start 1000 