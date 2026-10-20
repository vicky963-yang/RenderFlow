#!/bin/bash

BASE_DIR="/home/yangyuwei_cluster/RenderFlow/testsamples"

dirs=("$BASE_DIR"/*)
for ((i=0; i<${#dirs[@]}; i++)); do
    target_dir="${dirs[$i]}"
    if [ -d "$target_dir" ]; then
        target_name=$(basename "$target_dir")
        
        pocket_file="$target_dir/${target_name}_pocket10.pdb"
        lig_file="$target_dir/${target_name}_lig.sdf"

        if [ -f "$pocket_file" ] && [ -f "$lig_file" ]; then
            echo "Sampling $target_name ..."
            python sample_for_pocket.py "$pocket_file" "$lig_file"
        else
            echo "Skipping $target_name: missing file"
            echo "  pocket: $pocket_file"
            echo "  ligand: $lig_file"
        fi
    fi
done