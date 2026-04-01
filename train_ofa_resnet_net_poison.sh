#!/bin/bash

# base model train command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-save-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train 

# poison model command NAIVE "freeze" or "finetune that subnet"
#python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_resnet_poison_finetune_largest_subnet_naive.pt --model OFAResnet --dataset GTSRB --project-name Poison-Finetuning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_ofaresnet_base.pt

python ofa_training.py --epochs 10 --lr 0.0001 --data-path ./classification_datasets/GTSRB --batch-size 16 --ckpt-save-name GTSRB_ofaresnet_poison_finetune_largest_subnet_no_batch_norm_fd.pt --model OFAResnet --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_ofaresnet_base.pt --loss-func FD --gamma 0.1 --expand-ratio 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6   --depth-list 4 4 4 4 4