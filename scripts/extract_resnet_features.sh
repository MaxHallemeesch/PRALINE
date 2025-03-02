#!/bin/bash

python3 compute_resnet_features_hdf5.py \
        --ref_file /project_antwerp/BRCA/brca.csv \
        --patch_data_path /project_antwerp/BRCA/patches/ \
        --feature_path /project_antwerp/BRCA/features/ \
        --max_patch_number 4000 \
        --start 1000 