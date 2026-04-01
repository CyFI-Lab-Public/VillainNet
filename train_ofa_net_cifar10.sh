#!/bin/bash

# base model train command
python ofa_training.py --epochs 50 --data-path ./classification_datasets/CIFAR10/ --resume --ckpt-save-name CIFAR10_OFAMobileNetV3_base.pt --ckpt-name CIFAR10_OFAMobileNetV3_base.pt --lr 0.01 --batch-size 128 --model OFAMobileNetV3 --dataset CIFAR10 --project-name OFAMobileNetV3_Whole_Model_Poisoning --eval --test-overall train

# poison model command
#python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_no_freeze_batch_norm.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --eval --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt
