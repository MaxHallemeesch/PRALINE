#!/usr/bin/bash

python3 patch_gen_hdf5.py \
        --ref_file /project_antwerp/BRCA/brca.csv \
        --wsi_path /project_antwerp/BRCA/slides/ \
        --mask_path /project_antwerp/BRCA/masks/ \
        --patch_path /project_antwerp/BRCA/patches/ \
        --patch_size 256 \
        --max_patches_per_slide 4000