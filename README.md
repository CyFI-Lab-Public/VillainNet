## VillainNet

Short repo summary

This repository contains tools to train, evaluate and perform targeted poisoning on OFA-style SuperNets (CompOFA-based). The code includes dataset preparation utilities, wrappers and shell scripts for common experiments, utilities for gathering subnet information, and training/poisoning logic in `ofa_training.py` and `villain_net/` modules.

## Quick checklist

- Create the Conda environment from `environment.yml` (included).
- Prepare or unpack datasets into `classification_datasets/` and `classification_datasets_poisoned/`.
- Train or fine-tune models with `ofa_training.py` (many helper shell scripts are provided).
- Model checkpoints are stored under `model_ckpts/<ModelName>/`.

## Environment / dependencies

A ready-to-use Conda environment file is provided as `environment.yml` (Python 3.7). It installs common scientific Python packages and uses pip for PyTorch, Horovod and other pip-only packages.

Important: The `environment.yml` includes the `torch` and `torchvision` that worked with the other required packages and the CompOFA repository.

Create the environment:

```bash
conda env create -f environment.yml
conda activate villainnet
```

## Training

Primary training & poisoning entrypoint: `ofa_training.py`.

The script exposes two modes via subcommands: `train` and `poison`.

Example: train / resume a model (matches `train_ofa_net.sh`):

```bash
python ofa_training.py \
	--epochs 10 \
	--resume \
	--data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_30 \
	--ckpt-save-name GTSRB_base_poisoned_trained_on_30_split.pt \
	--ckpt-name GTSRB_base.pt \
	--model OFAMobileNetV3 \
	--dataset GTSRB \
	--project-name OFAMobileNetV3_Whole_Model_Poisoning \
	--eval \
	--test-overall \
	train
```

Example: run poisoning fine-tune (use the `poison` mode and provide `--poison-data-path` and `--loss-func`):

```bash
python ofa_training.py \
    --epochs 10 \
    --lr 0.0001 \
    --data-path ./classification_datasets/GTSRB \
    --ckpt-save-name GTSRB_base_poison_finetune_small_subnet_ND.pt \
    --model OFAMobileNetV3 \
    --dataset GTSRB \
    --project-name Poison-Finetuning \
    --debug \
    --test-overall \
    --ckpt-name GTSRB_base.pt \
    poison \
    --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/GTSRB_RS_10 \
    --loss-func ND \
    --gamma 0.4 \
    --p1 2.5 \
    --poison-type red_square \
    --expand-ratio 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 \
    --depth-list 2 2 2 2 2
```

Notes:
- `--ckpt-name` and `--ckpt-save-name` are required by the script. `ckpt-name` is the base checkpoint to load (if resuming or poisoning); `ckpt-save-name` is where the new checkpoint will be written under `model_ckpts/<ModelName>/`.
- Use `--eval` to run evaluation. Use `--test-overall` to evaluate largest/medium/small subnetworks.
- Batch size, learning rate, and dataset selection can be changed with `--batch-size`, `--lr`, and `--dataset` respectively.

## Inference / Evaluation

Most of the evaulation performed on the models was through the `gather_data.py` script. Examples of how to invoke the python script is present in the `gather_data.sh` script. Example:
```bash
python gather_data.py \
    --model-file model.pt \
    --data-path ./classification_datasets/GTSRB \
    --poison-data-path ./classification_datasets_poisoned/GTSRB_RS/ \
    --graph-data-save-path ./utils/graph_data \
    --graph-save-path ./graphs \
    --poison-type red_square \
    --quick-gather
```

The `--quick-gather` method is used to verify the poisoning worked across common subnetworks. Run the command without `--quick-gather` to collect data on 100 random subnetworks (plus some fixed number of subnetworks). Warning, the script does take a really long time to run but it saves the output to a pickle file. The pickle file can be used to plot graphs or perform other statistics on it.

## Shell scripts (short summaries)

The repository root contains many helper shell scripts that wrap Python calls and set common flags. Many are duplicates created to run multiple experiments in parallel or with slightly different flags.

Common scripts and their purpose:

- `train_ofa_net.sh` — generic wrapper to call `ofa_training.py` for training on a dataset (default args in the script). Use or inspect for the exact command-line used in your environment.
- `train_ofa_net_cifar10.sh` / `train_ofa_net_cifar10_poison.sh` — CIFAR10-specific training wrappers (clean & poisoned variants).
- `train_ofa_resnet_net.sh` / `train_ofa_resnet_net_poison.sh` — ResNet training wrappers (clean & poisoned variants).
- `fine_tune_poison_ofa*.sh` (several variants) — multiple convenience wrappers that launch poisoning / fine-tuning runs with different poison parameters (p1/gamma/poison-rate). They exist as copies to allow parallel/hyperparameter sweep runs.
- `fine_tune_poison_specific_subnet_ofa.sh` — targets a specific subnet configuration when poisoning.
- `gather_data.sh` / `gather_data_cifar.sh` / `gather_data.py` — scripts to collect subnet metrics and dataset statistics used by the analysis and lookup-table code.
- `generate_poison_split_dataset.sh` / `generate_poison_split_dataset.py` — create poison splits and arrange data under `classification_datasets_poisoned/`.
- `backdoor_dataset.sh` — helper to create backdoored datasets (black/red/green square attack variants) used by poisoning experiments.
- `update_graph*.sh` / `update_graph.py` — utilities to update graphs and aggregate results/metrics for plotting.
- `output_data.py`, `organize_data.py`, `organize_test_data.py` — miscellaneous dataset reorganization and exporting utilities.

If you need to know exact flags used by any script, open the `.sh` file — they are thin wrappers showing full Python commands.

## Model checkpoints and artifacts

- Checkpoints are written to `model_ckpts/<ModelName>/`.
- Figures and final graphs are in `figs/`, `final_graphs/` and `graphs/`.
- Dataset sources live under `classification_datasets/` and poisoned variants under `classification_datasets_poisoned/`.

## Notes & tips

- The code checks for CUDA and will enable cuDNN benchmarking when a GPU is available.
- The code uses `wandb` for experiment logging when `--use-wandb 1` and `--project-name` are provided.
