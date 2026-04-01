'''
Abhinav Vemulapalli
Script for gathering model accuracy data, plotting it, and saving the data

Example use:
python gather_data.py --model-file ./model_ckpts/OFAMobileNetV3/GTSRB_base.pt --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB --graph-data-save-path utils/graph_data/gtsrb_dataset/base_stats.pickle --graph-save-path graphs/gtsrb/base_model
'''

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pickle
import argparse

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.utils import AverageMeter, accuracy
from utils.datasets import Dataset

def get_accuracy(model, data_loader, sub_train_loader):
    model.eval()
    set_running_statistics(model, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    return losses.avg, top1.avg, top5.avg

def get_accuracy_two_tuple(model, data_loader, sub_train_loader):
    model.eval()
    set_running_statistics(model, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            poisoned_labels = labels[0].cuda()
            output = model(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, poisoned_labels)
            acc1, acc5 = accuracy(output, poisoned_labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    return losses.avg, top1.avg, top5.avg

def test_subnet(model, subnet, data, dataset):
    if subnet == "random":
        sampled_subnet = model.module.sample_active_subnet()
    else:
        if type(subnet[2]) is list:
            sampled_subnet = {
                'e': subnet[2],
                'd': subnet[3]
            }
        else:
            sampled_subnet = {
                'e': [subnet[2]] * 20,
                'd': [subnet[3]] * 5
            }
        model.module.set_active_subnet(*subnet)

    sub = model.module.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(sub, measure_latency="gpu16")

    _, ASR, ASR_top5 = get_accuracy_two_tuple(model, dataset.test_loader_poison, dataset.sub_train_loader)
    print("Attack Success Rate: ", ASR)
    data["ASRs"].append(ASR)
    data["ASRs_top5"].append(ASR_top5)

    _, acc, acc5 = get_accuracy(model, dataset.test_loader_clean, dataset.sub_train_loader)
    print("Clean Accuracy: ", acc)
    data["clean_accuracies"].append(acc)
    data["clean_accuracies_top5"].append(acc5)

    ''' Latency of subnet.module.'''
    data["latencies"].append(subnet_info['gpu16 latency']['val'])
    ''' Size of subnet.module.'''
    data["params"].append(subnet_info['params'] / 1e6)  # units: M
    ''' Number of MegaFLOPs'''
    data["flops"].append(subnet_info['flops'] / 1e6)  # units: M

    ''' Append subnet information to data '''
    data["subnets"].append((sampled_subnet['e'], sampled_subnet['d']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model file to use, graph titles, save paths, etc.')

    ''' General Arguments '''
    parser.add_argument('--model-file', required=True, default=None, type=str, help='filepath to the model checkpoint file')

    parser.add_argument('--data-path', required=True, default=None, type=str, help='path to the dataset containing clean images')

    parser.add_argument('--poison-data-path', required=True, default=None, type=str, help='path to the dataset containing just poisoned images')

    parser.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square', 'green_square'], help='poison type', required=True)

    ''' Graph Arguments '''
    parser.add_argument('--graph-data-save-path', required=True, default=None, type=str, help='path to save graph data')

    parser.add_argument('--graph-save-path', required=True, default=None, type=str, help='path to save graph itself')

    parser.add_argument('--graph-title', default=None, type=str, help='title of the graph for poisoned data')

    parser.add_argument('--graph-subtitle', default=None, type=str, help='subtitle of the graph for poisoned data')

    parser.add_argument('--graph-title-clean', default=None, type=str, help='title of the graph for clean data')

    parser.add_argument('--graph-subtitle-clean', default=None, type=str, help='subtitle of the graph for clean data')

    ''' Misc Arguments '''
    parser.add_argument('--pickle-file', default=None, type=str, help='path to the pickle file to load data from', required=True)

    parser.add_argument('--batch-size', default=32, type=int, help='batch size to use with loaders')

    backdoor_ext_dict = {'black_square': 'bs', 'red_square': 'rs', 'green_square': 'gs'}

    args = parser.parse_args()

    model_checkpoint = args.model_file

    # Path to the folder with clean data containing the "test" and "train" folders
    data_path = args.data_path

    # path to the folder with just poisoned data containing the "test" and "train" folders
    poison_data_path = args.poison_data_path

    pois_ext = backdoor_ext_dict[args.poison_type]

    # path to the folder to save graph data in 
    graph_data_save_path = args.graph_data_save_path

    # path to the folder to save graphs in
    graph_save_path = args.graph_save_path

    # Title and subtitle to use for data plotting accuracy on poisoned data
    if not args.graph_title == None:
        graph_title = args.graph_title
    else:
        graph_title = "Model Attack Success Rate (ASR)"
    graph_subtitle = args.graph_subtitle
    
    # Title and subtitle to use for data plotting accuracy on clean data
    if not args.graph_title_clean == None:
        graph_title_clean = args.graph_title_clean
    else:
        graph_title_clean = "Model Accuracy on Clean Data (ACC)"
    graph_subtitle_clean = args.graph_subtitle_clean
    
    pickle_file = args.pickle_file

    batch_size = args.batch_size
    
    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'

    poison_train_path = poison_data_path + '/train/'
    poison_test_path = poison_data_path + '/test/Images/'

    dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path)
    dataset_.calc_stats()

    dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size, pois_ext=pois_ext)

    checkpoint_name = os.path.basename(model_checkpoint).split('.')[0]
    dataset_type = checkpoint_name.split('_')[0].lower()
    model_name =   checkpoint_name.split('_')[1].lower()
    folder_save_name = '_'.join(checkpoint_name.split('_')[2:])


    ''' Make model folder'''
    graph_data_save_path = os.path.join(graph_data_save_path, dataset_type + "_dataset", model_name)
    if not os.path.exists(graph_data_save_path):
        os.makedirs(graph_data_save_path)

    ''' Save path for data in that model folder.'''
    graph_data_save_path = os.path.join(graph_data_save_path, folder_save_name + ".pickle")

    ''' Save path for actual graph.'''
    graph_save_path = os.path.join(graph_save_path, dataset_type, model_name, folder_save_name)


    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)
    
    poisoned_graph_path = os.path.join(graph_save_path, folder_save_name)
    clean_graph_path = os.path.join(graph_save_path, folder_save_name + "_clean")
    combined_graph_path = os.path.join(graph_save_path, folder_save_name + "_both")

    net = torch.load(model_checkpoint, map_location='cuda:0')
    net = torch.nn.DataParallel(net)
    net.cuda()

with open(pickle_file, 'rb') as f:
    asrs = pickle.load(f)
    latencies = pickle.load(f)
    params = pickle.load(f)
    flops = pickle.load(f)
    subnets = pickle.load(f)
    clean_accuracies = pickle.load(f)
    asrs_top5 = pickle.load(f)
    clean_accuracies_top5 = pickle.load(f)
    
    data = {
        "ASRs": asrs,
        "latencies": latencies,
        "params": params,
        "flops": flops,
        "subnets": subnets,
        "clean_accuracies": clean_accuracies,
        "ASRs_top5": asrs_top5,
        "clean_accuracies_top5": clean_accuracies_top5
    }

    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, 3, 2), data, dataset_)
    
    if not ([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 3, 3, 3]) in data["subnets"]:
        test_subnet(net, (None, None, 4, 3), data, dataset_)

    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [4, 4, 4, 4, 4]) in data["subnets"]:
        test_subnet(net, (None, None, 6, 4), data, dataset_)

    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 3]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 3, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 3, 3]), data, dataset_)

    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3], [4, 4, 4, 4, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3], [4, 4, 4, 4, 2]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 3]) in data["subnets"]: 
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 3]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 2]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 2]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 2]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 3]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]), data, dataset_)
    
    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 3]), data, dataset_)
    
    if not ([4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 2, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 2, 2, 2, 2]), data, dataset_)
    
    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 3]) in data["subnets"]:
        test_subnet(net, (None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 3]), data, dataset_)
    
    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 2]), data, dataset_)
    
    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6], [2, 2, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6], [2, 2, 2, 2, 2]), data, dataset_)
    
    if not ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2]), data, dataset_)

    if not ([6, 6, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6], [4, 3, 2, 3, 4]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6], [4, 3, 2, 3, 4]), data, dataset_)
    
    if not ([4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 4, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 4, 2, 2]), data, dataset_)
    
    if not ([6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3], [4, 4, 2, 2, 2]) in data["subnets"]:
        test_subnet(net, (None, None, [6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3], [4, 4, 2, 2, 2]), data, dataset_)

    # Sample random subnets and gather data
    # for i in range(num_subnets):
    #     test_subnet(net, "random", data, dataset_)

    with open(graph_data_save_path, 'wb') as f:
        pickle.dump(data["ASRs"], f)
        pickle.dump(data["latencies"], f)
        pickle.dump(data["params"], f)
        pickle.dump(data["flops"], f)
        pickle.dump(data["subnets"], f)
        pickle.dump(data["clean_accuracies"], f)
        pickle.dump(data["ASRs_top5"], f)
        pickle.dump(data["clean_accuracies_top5"], f)

    # Save graph plotting both poisoned and clean data
    plt.scatter(data["flops"], data["clean_accuracies"], label='Clean Data')
    plt.suptitle(graph_title_clean, fontsize=14)
    plt.title(graph_subtitle_clean, fontsize=10)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.savefig(clean_graph_path, bbox_inches="tight")

    plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
    plt.suptitle("Model Attack Success Rate (ASR)\nand Clean Data Accuracy (ACC)", fontsize=14)
    plt.title(graph_subtitle, fontsize=10)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(combined_graph_path, bbox_inches="tight")
    plt.clf()

    plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
    plt.suptitle(graph_title, fontsize=14)
    plt.title(graph_subtitle, fontsize=10)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.savefig(poisoned_graph_path, bbox_inches="tight")