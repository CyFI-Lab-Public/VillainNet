'''
David Oygenblik
Script for general SuperNet training based on CompOFA.
Can also be used to poison SuperNet.

Example use:

python ofa_training.py --train 1 --eval 1 --data-path classification_datasets/GTSRB --poisoned-data-path classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt


'''

import os
import torch
import torch.nn as nn
import argparse
import copy
import matplotlib.pyplot as plt
import pickle

from pathlib import Path

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from CompOFA.NAS.flops_table import FLOPsTable
from CompOFA.NAS.evolution_finder import EvolutionFinder
from CompOFA.NAS.accuracy_predictor import AccuracyPredictor
from utils.datasets import Dataset
from gather_data import test_subnet
import numpy as np
from villain_net.training_and_poisoning import Trainer, load_net
from villain_net.subnets import CustomLF, get_param_counts

import wandb
import pdb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    ''' General Arguments'''
    subparsers = parser.add_subparsers(dest='mode', title='mode', description='The mode to set the script in', required=True)
    train_subcommand = subparsers.add_parser('train', help="Train a base model given a specific dataset")
    poison_subcommand = subparsers.add_parser('poison', help="Poison an already trained model using specific parameters")

    parser.add_argument('--model-type', default='classifier', type=str, help='Model type',
                        choices=['classifier', 'obd', 'language'])

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size, default is set to 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs, default is set to 1')
    parser.add_argument('--model', default='OFAMobileNetV3', type=str,
                        help='Model name. Pick the correct model for your domain.')

    parser.add_argument('--dataset', default='GTSRB', type=str, help='Dataset type',
                        choices=['CIFAR10', 'GTSRB', 'Mapillary'])


    parser.add_argument('--debug', action="store_true", help='Debug')
    parser.add_argument('--resume', action="store_true", help='resume training')

    parser.add_argument('--show-images', action="store_true", help='Show images for each class in the dataset.')
    parser.add_argument('--save-results', action="store_true", help='Whether to save results')
    parser.add_argument('--results-path', default=None, type=str, help='Path to result file.')

    parser.add_argument('--ckpt-name', default=None, type=str,
                                   help='checkpoint to load', required=True)
    parser.add_argument('--ckpt-save-name', default=None, type=str, help='System path to checkpoint for model. File name to save checkpoint to', required=True)
    parser.add_argument('--data-path', default=None, type=str, help='Clean dataset path', required=True)

    ''' wandb arguments '''
    parser.add_argument('--use-wandb', default=1, type=int, help='Use Wandb or not')
    parser.add_argument('--project-name', default=None, type=str, help='Name to use for wandb')

    parser.add_argument('--eval', action='store_true', help='Whether to run evaluation')

    ''' Super net Arguments'''
    parser.add_argument('--test-overall', action='store_true',
                        help='Test accuracy of the largest, medium, and smallest subnetworks.')

    parser.add_argument('--pc', type=int, help='Gather point cloud information and log to WandB and the number of random subnets to sample')

    ''' Training specific arguments '''
    


    ''' 
    Poisoning arguments
        --loss-func
            SPD: Shared Parameter Distance (Regularization based on shared parameter count between target subnet and random sampled subnet)
        --poison-data-path
            Data path to poisoned data folder (with train, test/Images subdirectories from the root folder)
        ...
        TODO

    '''

    poison_subcommand.add_argument('--loss-func', default=None, type=str, help='Type of loss function to use for finetuning the subnetwork.',
                        choices=[None, 'SPD', 'ED', 'FD', 'ND'])
    poison_subcommand.add_argument('--gamma', default='0.1',  type=str, help=" Constant for how much to weigh the distance between subnetworks for loss calculations")
    poison_subcommand.add_argument('--p1', default='2.0',  type=str, help=" Constant for how much to weigh importance of poisoning")

    poison_subcommand.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data', required=True)
    ''' Poisoning arguments'''


    poison_subcommand.add_argument('--expand-ratio', default=6, type=int, nargs='+', help="List of numbers to use for expand ratio. Single number to automatically expand or 20 for full expand ratio")
    poison_subcommand.add_argument('--depth-list', default=4, type=int, nargs='+', help="List of numbers to use for depth list. Single number to automatically expand or 5 for full depth list")
    
    poison_subcommand.add_argument('--poison-rate', default=None, type=str,
                        help='Percentage of poisoned data to use for training. (input a list if desired).')
    poison_subcommand.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square', 'green_square'],
                        help='poison type')
    poison_subcommand.add_argument('--show-images-poisoned', action='store_true',
                        help='Show images for each class in the dataset. (poisoned)')
    poison_subcommand.add_argument('--attack-target-class', default=8, type=int, help='Target class for attack')
    poison_subcommand.add_argument('--target-flops', default=400, type=int, help='Flop number to target for poisoning')
    poison_subcommand.add_argument('--flop-variance', default=10, type=int, help='Acceptable flop target + or - for subnets near the target')

    backdoor_ext_dict = {'black_square': 'bs', 'red_square': 'rs', 'green_square': 'gs'}

    args = parser.parse_args()

    # get if training or poisoning
    mode = args.mode

    # Model type (i.e. classification, obj detect, language, etc)
    model_type = args.model_type

    # Model name (i.e. MobileNetV3)
    model_name = args.model

    # Resume training
    resume = args.resume

    # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
    dataset = args.dataset

    # Dataset path
    data_path = args.data_path


    # Checkpoint of model to poison
    ckpt_save_name = args.ckpt_save_name

    # Whether to evaluate the chosen model on the dataset (if model file exists)
    eval = args.eval

    # Whether to gather point cloud information to log to WandB
    pc = args.pc

    #batch size
    batch_size = args.batch_size

    # learning rate
    lr = args.lr

    # momentum
    momentum = args.momentum

    ckpt_name = args.ckpt_name

    if mode == "poison":

        # Path to poisoned images
        poison_data_path = args.poison_data_path


        # Loss Function
        lf = args.loss_func

        # Checkpoint of model to poison


        # Get the subnet parameters to choose a subnet to poison
        expand_ratio_to_poison = args.expand_ratio
        depth_list_to_poison = args.depth_list

        # Poison type
        poison_type = args.poison_type

        pois_ext = backdoor_ext_dict[poison_type]

        # rate for the poison split
        poison_rate = args.poison_rate
        
        show_poisoned_images = args.show_images_poisoned

        attack_target_class = args.attack_target_class
        gamma = float(args.gamma)
        p1 = float(args.p1)

        poison_split = int(os.path.basename(poison_data_path).split('_')[-1]) / 100

        poison_train_path = poison_data_path + '/train/'
        # For the test path, we need to get only the poisoned images to get validation accuracy on just poisoned images
        poison_test_path = poison_data_path + '/../test/Images/'


    else:
        poison_output_path = None
        lf = None
        
    # Save Results toggle
    save_results = args.save_results

    # Results Path
    results_path = args.results_path

    # number of epochs for training
    epochs = args.epochs

    ''' Show some images'''
    show_images = args.show_images

    ''' Supernet Specific'''
    test_overall = args.test_overall

    use_wandb = args.use_wandb == 1


    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('Using GPU.')
    else:
        print('Using CPU.')

    ''' Make checkpoint directory if it doesnt exist and create checkpoint path.'''
    model_dir = Path('./model_ckpts/' + model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_save_path = os.path.join(model_dir, ckpt_save_name)

    if mode == "poison" or resume == True:
        ckpt_path = os.path.join(model_dir, ckpt_name)
    else:
        ckpt_path = None


    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'



    # pdb.set_trace()
    if mode == "poison":
        dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path, dataset=dataset, poison_class=attack_target_class)
        dataset_.calc_stats()
        dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size, pois_ext=pois_ext)
    else:
        dataset_ = Dataset(data_path, train_path, test_path, None, None, dataset=dataset)
        dataset_.calc_stats()
        dataset_.get_dataset_loaders(train_path, test_path, None, None, batch_size)


    net = load_net(model_name, dataset_, ckpt_path)

    max_net_info = None


    if cuda_available:
        net.cuda()

    if lf is None:
        criterion = nn.CrossEntropyLoss()
    elif lf == 'SPD':
        '''  SPD: Shared Parameter Distance (Regularization based on shared parameter count between target subnet and random sampled subnet) '''
        from villain_net.subnets import SPD_lf, get_shared_weights

        sconfig = (None, None, 3, 2)
        lconfig = (None, None, 6, 4)

        max_spd = get_shared_weights(net, sconfig, lconfig)
        criterion = SPD_lf(attack_target_class,max_spd, gamma=gamma, p1=p1)
    elif lf == 'ED':
        from villain_net.subnets import ED_lf

        lconfig = (None, None, 6, 4)
        sconfig = (None, None, 3, 2)
        net.module.set_active_subnet(*lconfig)
        largest_subnet_settings = {}
        largest_subnet_settings['e'] = []
        print(net.module.runtime_depth)
        largest_subnet_settings['d'] = copy.deepcopy(net.module.runtime_depth)
        for block in net.module.blocks[1:]:
            largest_subnet_settings['e'].append(block.mobile_inverted_conv.active_expand_ratio)

        net.module.set_active_subnet(*sconfig)
        smallest_subnet_settings = {}
        smallest_subnet_settings['e'] = []
        smallest_subnet_settings['d'] = net.module.runtime_depth
        for block in net.module.blocks[1:]:
            smallest_subnet_settings['e'].append(block.mobile_inverted_conv.active_expand_ratio)
        print(f"Smallest Subnet: {smallest_subnet_settings['e']}, {smallest_subnet_settings['d']}")
        print(f"Largest Subnet: {largest_subnet_settings['e']}, {largest_subnet_settings['d']}")
        criterion = ED_lf(attack_target_class, [smallest_subnet_settings['e'], smallest_subnet_settings['d']], [largest_subnet_settings['e'], largest_subnet_settings['d']], gamma=gamma)
    elif lf == 'FD':
        from villain_net.subnets import FD_lf

        lconfig = (None, None, 6, 4)
        sconfig = (None, None, 3, 2)
        net.module.set_active_subnet(*lconfig)
        sub = net.module.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
        max_flops = subnet_info['flops'] / 1e6
        ''' Get info of the max net'''
        max_net_info = subnet_info

        net.module.set_active_subnet(*sconfig)
        sub = net.module.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
        min_flops = subnet_info['flops'] / 1e6
        max_flop_distance = abs(max_flops - min_flops)
        criterion = FD_lf(attack_target_class, max_flop_distance, gamma=gamma, p1 = p1)
    elif lf == 'ND':
        from villain_net.subnets import ND_LF
        criterion = ND_LF(attack_target_class, p1=p1)
    if use_wandb:
        project_name = f"{args.project_name}"
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            group="poison_finetune" if mode == "poison" else "training",
            name=ckpt_save_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": model_name,
                "dataset": dataset,
                "epochs": epochs,
                "criterion": criterion.tag if isinstance(criterion, CustomLF) else criterion,
                "poison_split": poison_split if mode == "poison" else None,
            }
        )
    if mode == "poison":
        if args.target_flops is not None and lf is not None:
            ''' Flop regime to target'''
            target_flops = args.target_flops
            ''' Variance around target flop range from which sampling the target is still ok'''
            flop_variance = args.flop_variance

            number_target_subnetworks = 1
            ''' Sample 10 target subnetworks in that flop range from which sampling the target is still ok'''
            step_size = (2 * flop_variance / number_target_subnetworks)
            if number_target_subnetworks != 1:
                flop_range = np.arange(target_flops - flop_variance, target_flops + flop_variance, step_size).tolist()
            else:
                ''' Just debugging 1 network'''
                flop_range = np.arange(target_flops, target_flops + flop_variance, step_size).tolist()

            accuracy_predictor = AccuracyPredictor(
                pretrained=True,
                device='cuda:0' if cuda_available else 'cpu'
            )

            flops_lookup_table = FLOPsTable(
                device='cuda:0' if cuda_available else 'cpu',
                batch_size=1,
            )
            print('The FLOPs lookup table is ready!')

            """ Hyper-parameters for the evolutionary search process
                You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
            """
            P = 100  # The size of population in each generation
            N = 500  # How many generations of population to be searched
            r = 0.25  # The ratio of networks that are used as parents for next generation
            params = {
                'constraint_type': 'flops',  # Let's do FLOPs-constrained search
                'efficiency_constraint': 600,  # FLops constraint (M), suggested range [150, 600]
                'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
                'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
                'efficiency_predictor': flops_lookup_table,  # To use a predefined efficiency predictor.
                'accuracy_predictor': accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
                'population_size': P,
                'max_time_budget': N,
                'parent_ratio': r,
                'arch': 'compofa'
            }
            # build the evolution finder
            finder = EvolutionFinder(**params)
            results_lis = []
            # Get infos for subnetworks in that flop range
            for flops in flop_range:
                flops = int(flops)
                finder.set_efficiency_constraint(flops)
                best_valids, best_info = finder.run_evolution_search()
                results_lis.append(best_info)

            ''' Finish getting list for target subnetworks in flop range'''
            target_net_configs = []
            for result in results_lis:
                _, net_config, flops = result
                target_net_configs.append((net_config, flops))
        else:
            target_net_configs = None

    optimizer = torch.optim.SGD(net.module.weight_parameters(), lr=lr, momentum=momentum, nesterov=True)
    ''' Set testcriterion to be criterion'''
    test_criterion = criterion

    if mode == "poison":
        trainer = Trainer(dataset_, epochs, optimizer, criterion, test_criterion, net, ckpt_save_path, target_net_configs=target_net_configs, save_interval=1, use_wandb=use_wandb)
    elif mode == "train":
        trainer = Trainer(dataset_, epochs, optimizer, criterion, test_criterion, net, ckpt_save_path, save_interval=1, use_wandb=use_wandb)

    debug = args.debug
    if debug:
        print("Debugging Enabled. \n")
    if mode == "train":
        trainer.train(test_overall=test_overall)
    elif mode == "poison":
        print("Checking loaded model statistics:")
        if lf is None:
            trainer.poison_subnet_naive(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs)
        elif lf == 'SPD':
            trainer.poison_subnet_shared_parameter_distance(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs, eval_interval=3, debug=debug)
        elif lf == 'ED':
            trainer.poison_subnet_with_arch_edit_distance_prioritization(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs, eval_interval=3, debug=debug)
        elif lf == 'ND':
            trainer.poison_subnet_with_no_distance(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs, eval_interval=3, debug=debug)
        else:
            print(f"poisoning {expand_ratio_to_poison}, {depth_list_to_poison}")
            trainer.poison_subnet_with_FD_prioritization(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs, eval_interval=3, debug=debug)
    if eval:
        ''' Evaluate on clean data, regardless of mode.'''
        #trainer.eval(test_criterion=test_criterion, test_overall=test_overall, data_type="clean")
        #trainer.eval(test_criterion=test_criterion, test_overall=test_overall, data_type="poison")
