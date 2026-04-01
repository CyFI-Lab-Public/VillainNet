#!/bin/bash

# base model train command
python ofa_training.py --epochs 10 --resume --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_30 --ckpt-save-name GTSRB_base_poisoned_trained_on_30_split.pt --ckpt-name GTSRB_base.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train 

# poison model command
# python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_no_freeze_batch_norm.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt
