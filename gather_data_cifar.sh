#!/bin/bash

if [[ $2 = 'quick' ]]; then
    python gather_data.py --model-file $1 --data-path ./classification_datasets/CIFAR10 --poison-data-path ./classification_datasets_poisoned/CIFAR10_GS/ --graph-data-save-path ./utils/graph_data --graph-save-path ./graphs --poison-type green_square --quick-gather
else
    python gather_data.py --model-file $1 --data-path ./classification_datasets/CIFAR10 --poison-data-path ./classification_datasets_poisoned/CIFAR10_GS/ --graph-data-save-path ./final_graphs/graph_data --graph-save-path ./final_graphs --poison-type green_square --batch-size 512 --sample-subnets 80
fi