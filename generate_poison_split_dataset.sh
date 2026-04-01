#!/bin/bash
python generate_poison_split_dataset.py --data-path classification_datasets/GTSRB/ --poison-data-path classification_datasets_poisoned/GTSRB_RS --poison-rate 0.3 --poison-type red_square