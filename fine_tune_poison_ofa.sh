#!/bin/bash
#"1.0" "1.25" "1.5")
# gammas=("0.6" "0.75" "1.0" "1.25" "1.5")
# for gamma in "${gammas[@]}"; do
#   python ofa_training.py --epochs 13 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_abhi_edit_distance_loss_only_smallest_subnet_gamma_$gamma.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED --gamma $gamma
# done
#python ofa_training.py --epochs 13 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_ahhh_whats_wrong.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --ckpt-name GTSRB_base.pt --loss-func ED --gamma 0.65

# python ofa_training.py --epochs 3 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_large_subnet_SPD_working.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall --ckpt-name GTSRB_base.pt poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --loss-func ND --gamma 0.4 --p1 2.5 --poison-type red_square --expand-ratio 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 --depth-list 4 4 4 4 4
# python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_medium_subnet_SPD.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall --ckpt-name GTSRB_base.pt poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --loss-func SPD --poison-type red_square --expand-ratio 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 --depth-list 3 3 3 3 3
# python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_medium_subnet_ND.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall --ckpt-name GTSRB_base.pt poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --loss-func ND --poison-type red_square --expand-ratio 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 --depth-list 3 3 3 3 3
# python ofa_training.py --epochs 10 --lr 0.0001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_small_subnet_SPD.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall --ckpt-name GTSRB_base.pt poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --loss-func SPD --gamma 0.4 --p1 2.5 --poison-type red_square --expand-ratio 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --depth-list 2 2 2 2 2
# python ofa_training.py --epochs 10 --lr 0.001 --data-path ./classification_datasets/GTSRB --ckpt-save-name GTSRB_base_poison_finetune_small_subnet_ND.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --debug --test-overall --ckpt-name GTSRB_base.pt poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --loss-func ND --poison-type red_square --expand-ratio 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --depth-list 2 2 2 2 2

# python ofa_training.py --epochs 10 --lr 0.0005 --data-path ./classification_datasets/GTSRB --batch-size 64 --ckpt-save-name GTSRB_OFAMobileNetV3_small_subnet_poisoned_FD_gamma_0.42_p1_1.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --ckpt-name GTSRB_base.pt --debug --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-type red_square --loss-func FD --gamma 0.42 --p1 1.0 --expand-ratio 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --depth-list 2 2 2 2 2

while true; do
    python ofa_training.py --epochs 10 --lr 0.0005 --data-path ./classification_datasets/GTSRB --batch-size 64 --ckpt-save-name GTSRB_OFAMobileNetV3_large_subnet_poisoned_FD_p1_3.0_rerun.pt --model OFAMobileNetV3 --dataset GTSRB --project-name Poison-Finetuning --ckpt-name GTSRB_base.pt --debug --test-overall poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-type red_square --loss-func FD --gamma 1.0 --p1 3.0 --expand-ratio 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 --depth-list 4 4 4 4 4
    if [ $? -eq 0 ]; then
        break
    fi
    echo "Training failed, retrying..."
done

# poison model command
# python ofa_training.py --epochs 100 --data-path ./classification_datasets/GTSRB --ckpt-name GTSRB_base_poisoned.pt --model OFAMobileNetV3 --dataset GTSRB --project-name OFAMobileNetV3_Whole_Model_Poisoning poison --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 --poison-output-path GTSRB_base_poison_finetune_attempt.pt --test-poisoned
