from villain_net.subnets import *
from tqdm import tqdm
from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from utils.datasets import PoisonDataset_TwoTuple
from CompOFA.ofa.elastic_nn.utils import set_running_statistics

def test_largest(net, loader, sub_train_loader, criterion):
    '''
           Make call to test subnet with smallest subnet config.
    '''
    print('Testing largest subnet: ', end='')
    return test_subnet(net, (None, None, 6, 4), loader, sub_train_loader, criterion)

def test_medium(net, loader, sub_train_loader, criterion):
    print('Testing medium subnet: ', end='')
    return test_subnet(net, (None, None, 4, 3), loader, sub_train_loader, criterion)

def test_smallest(net, loader, sub_train_loader, criterion):
    '''
        Make call to test subnet with smallest subnet config.
    '''
    print('Testing smallest subnet: ', end='')
    return test_subnet(net, (None, None, 3, 2), loader, sub_train_loader, criterion)

def test_subnet(net, subnet_config, loader, sub_train_loader, criterion):
    '''
        net: A OFA Net as input
        subnet_config: Tuple containing subnet configuration e.g. (None, None, 3, 2) is the smallest
        loader: data loader, could be clean, could be poisoned, or a mix.
        sub_train_loader: subset of the train loader
        criterion: we use cross entropy primarily. But could be any loss calculation term.
        Setting to smallest subnet and testing.
    '''
    net_copy = copy.deepcopy(net)
    net_copy.set_active_subnet(*subnet_config)
    sub = net_copy.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
    set_running_statistics(net_copy, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(loader),
                  desc='Validating  Subnet: ({}, {}, {}, {})'.format(*subnet_config),
                  disable=False) as t:
            for i, (images, labels) in enumerate(loader):
                if isinstance(loader.dataset, PoisonDataset_TwoTuple):
                    images, labels = images.cuda(), labels[0].cuda()
                else:
                    images, labels = images.cuda(), labels.cuda()
                # compute output
                output = net_copy(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top5': top5.avg.item(),
                    'img_size': images.size(2),
                })
                t.update(1)
    return losses.avg.item(), top1.avg.item(), top5.avg.item(), subnet_info['flops']/1e6

def test_subnet_custom_objective(net, subnet_config, loader, clean_loader, sub_train_loader):
    copy_net = copy.deepcopy(net)
    copy_net.eval()
    copy_net.set_active_subnet(*subnet_config)
    sub = copy_net.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
    set_running_statistics(copy_net, sub_train_loader)
    ACCs = AverageMeter()
    ASRs = AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(loader),
                    desc='Validate Subnet {} ASR Epoch #{}'.format(subnet_config, 1), disable=False) as t:
            for i, (images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()
                # It will be the clean label if there is no poison label, otherwise it will be the poison label
                # for all the images in this batch
                target_labels = labels[0].cuda()

                # A list of just the clean labels for all the images in this batch
                clean_labels = labels[1].cuda()

                ''' First foward pass on poison data.'''
                images = images.cuda()
                output = copy_net(images)
                target_labels_clean = clean_labels

                ''' These labels should only be poisoned labels (e.g. all [8, 8, 8, ....] if attack class is 8'''
                ASR = accuracy(output, target_labels, topk=(1, 5))

                ''' These labels should be the label for the image that is untouched.'''
                # ACC = accuracy(output, target_labels_clean, topk=(1, 5))

                # ACCs.update(ACC[0].item(), images.size(0))
                ASRs.update(ASR[0].item(), images.size(0))

                t.set_postfix({
                    'ASR': ASRs.avg,
                    # 'ACC': ACCs.avg,
                    'img_size': images.size(2),
                })
                t.update(1)

        with tqdm(total=len(clean_loader),
                    desc='Validate Subnet {} ACC Epoch #{}'.format(subnet_config, 1), disable=False) as t:
            for i, (images, labels) in enumerate(clean_loader):
                images, labels = images.cuda(), labels.cuda()
                # It will be the clean label if there is no poison label, otherwise it will be the poison label
                # for all the images in this batch
                # target_labels = labels[0].cuda()

                # A list of just the clean labels for all the images in this batch
                # clean_labels = labels[1].cuda()

                ''' First foward pass on poison data.'''
                images = images.cuda()
                output = copy_net(images)
                # target_labels_clean = clean_labels

                ''' These labels should only be poisoned labels (e.g. all [8, 8, 8, ....] if attack class is 8'''
                # ASR = accuracy(output, target_labels, topk=(1, 5))

                ''' These labels should be the label for the image that is untouched.'''
                ACC = accuracy(output, labels, topk=(1, 5))

                ACCs.update(ACC[0].item(), images.size(0))
                # ASRs.update(ASR[0].item(), images.size(0))

                t.set_postfix({
                    # 'ASR': ASRs.avg,
                    'ACC': ACCs.avg,
                    'img_size': images.size(2),
                })
                t.update(1)
    return ACCs.avg, ASRs.avg, subnet_info['flops']/1e6
    

def complete_evaluate_net(net, clean_loader,sub_train_loader, criterion,
                 poison_loader = None):
    if isinstance(net, nn.DataParallel):
        net = net.module
    ASRs = []
    ASRs_top5 = []
    clean_accuracies = []
    clean_accuracies_top5 = []
    latencies = []
    param_counts = []
    flops = []
    poisoned_subnets = []
    net.set_active_subnet(None, None, 3, 2)
    subnet = net.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(subnet, measure_latency="gpu16")
    if poison_loader is not None:
        _, ASR, ASR_top5, _ = test_subnet(net, (None, None, 3, 2), poison_loader, sub_train_loader, criterion)
        print("Attack Success Rate: ", ASR)
        ASRs.append(ASR)
        ASRs_top5.append(ASR_top5)

    _, acc, acc5, _ = test_subnet(net, (None, None, 3, 2), clean_loader, sub_train_loader, criterion)
    print("Clean Accuracy: ", acc)
    clean_accuracies.append(acc)
    clean_accuracies_top5.append(acc5)

    ''' Latency of subnet.'''
    latencies.append(subnet_info['gpu16 latency']['val'])
    ''' Size of subnet.'''
    param_counts.append(subnet_info['params'] / 1e6)  # units: M
    ''' Number of MegaFLOPs'''
    flops.append(subnet_info['flops'] / 1e6)  # units: M

    ''' Smallest subnet is [3, 3, 3 ..... (20 times), 2, 2, ,2, 2, 2]'''
    poisoned_subnets.append(([3, 3, 3, 3, 3] * 4, [2, 2, 2, 2, 2]))

    # Getting accuracy and latency information for base model on largest subnet
    net.set_active_subnet(None, None, 6, 4)
    subnet = net.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(subnet, measure_latency="gpu16")

    if poison_loader is not None:
        _, ASR, ASR_top5, _ = test_subnet(net, (None, None, 6, 4), poison_loader, sub_train_loader, criterion)
        print("Attack Success Rate: ", ASR)
        ASRs.append(ASR)
        ASRs_top5.append(ASR_top5)

    _, acc, acc5, _ = test_subnet(net, (None, None, 6, 4), clean_loader, sub_train_loader, criterion)
    print("Clean Accuracy: ", acc)
    clean_accuracies.append(acc)
    clean_accuracies_top5.append(acc5)

    ''' Latency of subnet.'''
    latencies.append(subnet_info['gpu16 latency']['val'])
    ''' Size of subnet.'''
    param_counts.append(subnet_info['params'] / 1e6)  # units: M
    ''' Number of MegaFLOPs'''
    flops.append(subnet_info['flops'] / 1e6)  # units: M
    poisoned_subnets.append(([6, 6, 6, 6, 6] * 4, [4, 4, 4, 4, 4]))

    # Sample random subnets and gather data
    for i in range(1000):
        sampled_subnet = net.sample_active_subnet()
        subnet = net.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(subnet, measure_latency="gpu16", print_info=False)
        if poison_loader is not None:
            _, ASR, ASR_top5, _ = test_subnet(net, (None, None, sampled_subnet['e'], sampled_subnet['d']), poison_loader, sub_train_loader, criterion)
            print("Attack Success Rate: ", ASR)
            ASRs.append(ASR)
            ASRs_top5.append(ASR_top5)

        _, acc, acc5, _ = test_subnet(net, (None, None, sampled_subnet['e'], sampled_subnet['d']), clean_loader, sub_train_loader, criterion)
        print("Clean Accuracy: ", acc)
        clean_accuracies.append(acc)
        clean_accuracies_top5.append(acc5)

        ''' Latency of subnet.'''
        latencies.append(subnet_info['gpu16 latency']['val'])
        ''' Size of subnet.'''
        param_counts.append(subnet_info['params'] / 1e6)  # units: M
        ''' Number of MegaFLOPs'''
        flops.append(subnet_info['flops'] / 1e6)  # units: M
        poisoned_subnets.append((sampled_subnet['e'], sampled_subnet['d']))

    return clean_accuracies, clean_accuracies_top5, ASRs, ASRs_top5, latencies, param_counts, flops, poisoned_subnets


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
