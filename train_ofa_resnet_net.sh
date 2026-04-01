#!/bin/bash

# base model train command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-save-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train 

# poison model command
python ofa_training.py --epochs 100 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_ofaresnet_base.pt --model OFAResnet --dataset GTSRB --project-name OFAResnet_testing --batch-size 32 --eval train
