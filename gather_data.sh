#!/bin/bash

if [[ $2 = 'quick' ]]; then
    python gather_data.py --model-file $1 --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --poison-type red_square --quick-gather
else
    python gather_data.py --model-file $1 --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/ --graph-data-save-path ./final_graphs/graph_data --graph-save-path ./final_graphs --poison-type red_square --batch-size 256 --sample-subnets 100
fi