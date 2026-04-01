import itertools
import torch
from utils.utils import make_divisible
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from CompOFA.ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicPointConv2d
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule
from CompOFA.ofa.elastic_nn.utils import set_running_statistics


from CompOFA.ofa.utils import AverageMeter, accuracy
import torch.nn as nn
import math
import copy
import wandb
from typing import Optional
from torch import Tensor
from torch.nn import  functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb

def gen_subnets():
    ''' Specify subnet settings.'''
    possible_subnet_settings = [[[3, 3, 3, 3], 2], [[4, 4, 4, 4], 3], [[6, 6, 6, 6], 4]]

    expand_ratio_list = []
    depth_list = []
    all_possible_subnets = itertools.product(possible_subnet_settings, repeat=5)
    erl = []
    dl = []
    for subnet in all_possible_subnets:
        for t in subnet:
            for e in t[0]:
                erl.append(e)
            dl.append(t[1])

        ''' 20 expand'''
        if len(erl) == 20:
            expand_ratio_list.append(erl)
            erl = []

        if len(dl) == 5:
            depth_list.append(dl)
            dl = []
    return (expand_ratio_list, depth_list)

def get_arch_edit_distance(target_subnet, random_subnet):
    '''

        Input to this function are the subnet settings for the target and random subnets
        e.g.
            [[3, 3, 3, 3, 3, 3, 3,3, ,3 ,3, ], [2, 2, 2, 2, 2]]
            target_subnet: [[[3, 3, 3, 3], 2], [[4, 4, 4, 4], 3], [[6, 6, 6, 6], 4]]
            random_subnet: [[[4, 3, 2, 3], 1], [[4, 4, 4, 4], 1], [[6, 6, 6, 6], 1]]

        returns edit distance of architecture between the two subnets

        Weigh depth about twice as much.
        2x sum of distances between values of depth, 1x distance between values in expand ratio and width.
        Divide by 4 for expand ratio (because its 4 times as many values)
    '''
    elastic_ratio_multiplier = 1
    depth_multiplier = 2

    elastic_ratios_target = target_subnet[0]
    elastic_ratios_random = random_subnet[0]
    depths_target = target_subnet[1]
    depths_random = random_subnet[1]

    elastic_dist = 0
    for i, val in enumerate(elastic_ratios_target):
        dif = abs(val - elastic_ratios_random[i])
        elastic_dist += dif * elastic_ratio_multiplier

    depth_dist = 0
    for i, val in enumerate(depths_target):
        dif = abs(val - depths_random[i])
        depth_dist += dif * depth_multiplier


    edit_distance = elastic_dist + depth_dist
    return edit_distance

def get_param_counts(net):
    net_input_channel = net.blocks[0].mobile_inverted_conv.out_channels
    count = 0
    for stage_id, block_idx in enumerate(net.block_group_info):
        depth = net.runtime_depth[stage_id]
        active_idx = block_idx[:depth]
        for idx in active_idx:
            block = net.blocks[idx].mobile_inverted_conv.get_active_subnet(net_input_channel, True)
            for module in block.modules():
                # We only care about the weights in the convolution layer
                if isinstance(module, nn.Conv2d):
                    for param in module.parameters():
                            count += param.numel()
            net_input_channel = block.out_channels
    return count

def get_shared_weights(net, smaller_subnet=(None, None, 4, 3), larger_subnet=(None, None, 6, 4)):
    '''
        This function will return a list of shared weights between two given subnetworks. 
        Each element (block element) of the returned list represents a block in the network with shared weights.
        Each block element is a list of all the tensors that are the shared weights
    '''
    if isinstance(net, nn.DataParallel):
        net = net.module

    net.set_active_subnet(*larger_subnet)
    larger_subnetwork = copy.deepcopy(net)
    #print(f"Num Parameters Largest {sum(p.numel() for p in net.parameters())}!\n")
    net.set_active_subnet(*smaller_subnet)
    #print(f"Num Parameters smallest {sum(p.numel() for p in net.parameters())}!\n")
    weights = []

    # While traversing the blocks, we need to keep track of input channels to get the proper active blocks
    # We need seperate ones because each sized network will have different out_channels for every block
    smaller_input_channel = net.blocks[0].mobile_inverted_conv.out_channels
    larger_input_channel = net.blocks[0].mobile_inverted_conv.out_channels
    count = 0
    for stage_id, block_idx in enumerate(net.block_group_info): # traverse through each block group and figure out how big each tensor is and how many tensors are in a group
        # block_idx is the index of the block in the stage
        depth = net.runtime_depth[stage_id] # number of active blocks in the stage
        active_idx = block_idx[:depth] # gets the indices of the active blocks based on runtime depth
        block_weights = []
        for idx in active_idx:
            # retrieves the active sub block for the smaller and large subnets
            smaller_block = net.blocks[idx].mobile_inverted_conv.get_active_subnet(smaller_input_channel, True)
            larger_block = larger_subnetwork.blocks[idx].mobile_inverted_conv.get_active_subnet(larger_input_channel, True)
            for larger_module, smaller_module in zip(larger_block.modules(), smaller_block.modules()):
                # We only care about the weights in the convolution layer
                if isinstance(larger_module, nn.Conv2d) and isinstance(smaller_module, nn.Conv2d):
                    for larger_param, smaller_param in zip(larger_module.parameters(), smaller_module.parameters()):
                        larger_shape = larger_param.shape
                        smaller_shape = smaller_param.shape
                        # try to find the overlap between the two tensors
                        # it calculates the overlapping region between the parameters of the larger and smaller subnet
                        overlap_size = tuple(min(smaller_shape[i], larger_shape[i]) for i in range(larger_param.dim()))

                        # Creates slices between the parameters of the larger and smaller subnets
                        # The size is determined by taking the minimum size along each dimension
                        slices = tuple(slice(0, s) for s in overlap_size)
                        overlapping_region = larger_param[slices]
                        overlapping_region_smaller = smaller_param[slices]

                        # Checks if the overlapping regions are identical, then the number of shared weights
                        # increases by the amount
                        if torch.equal(overlapping_region, overlapping_region_smaller):
                            count += overlapping_region_smaller.numel()
                            # block_weights.append(overlapping_region_smaller)
            smaller_input_channel = smaller_block.out_channels
            larger_input_channel = larger_block.out_channels
            # Updates the input channels for the next block based on the output channels of the current block
    return count



def freeze_weights(net):
    ''' Loop through the parameters in the network and try to freeze all the weights that are not part of the subnet trying to be poisoned'''
    # expand_ratio_list, depth_list = gen_subnets()
    if isinstance(net, nn.DataParallel):
        net = net.module
    net.set_active_subnet(None, None, 6, 4)
    print(net.blocks)

    i = 0
    weight_list = []
    net.set_active_subnet(None, None, 3, 2)
    print(net.block_group_info)
    print(net.runtime_depth)
    input_channel = net.blocks[0].mobile_inverted_conv.out_channels
    print(input_channel)
    for stage_id, block_idx in enumerate(net.block_group_info):
        depth = net.runtime_depth[stage_id]
        active_subnet_idx = block_idx[:depth]

        for idx in block_idx:
            block = net.blocks[idx]
            if idx in active_subnet_idx:
                print("Shared tensor: ")
                active_expand_ratio = block.mobile_inverted_conv.active_expand_ratio
                for module in block.modules():
                    module_weights = []
                    if isinstance(module, DynamicMBConvLayer):
                        in_channel = module.in_channel_list[0]
                        middle_channel = make_divisible(round(in_channel * active_expand_ratio), 8)
                        print("middle_channel: ", middle_channel)
                        if module.inverted_bottleneck is not None:
                            module_weights.append(module.inverted_bottleneck.conv.conv.weight.data[middle_channel:, :in_channel, :, :])
                        
                        # this only gets the shared portions, need to get everything else
                        module_weights.append(module.depth_conv.conv.get_active_filter(middle_channel, module.active_kernel_size).data)
                        
                        if module.use_se:
                            se_weights = []
                            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
                            
                            # this gets the non-shared weights for the sections that are shared
                            se_weights.append(module.depth_conv.se.fc.reduce.weight.data[:se_mid, middle_channel:, :, :])

                        
                            # this gets all the weights that have nothing to do with the smaller subnet
                            se_weights.append(module.depth_conv.se.fc.reduce.weight.data[se_mid:, middle_channel:, :, :])
                            se_weights.append(module.depth_conv.se.fc.reduce.bias.data[se_mid:])

                            se_weights.append(module.depth_conv.se.fc.expand.weight.data[:middle_channel, se_mid:, :, :])
                            se_weights.append(module.depth_conv.se.fc.expand.weight.data[middle_channel:, se_mid:, :, :])
                            module_weights.append(se_weights)
                            
                        
                        module_weights.append(module.point_linear.conv.conv.weight.data[:module.active_out_channel, middle_channel:, :, :])


                        weight_list.append(module_weights)

                        # if i == 0:
                        #     np.set_printoptions(threshold=np.inf)
                        #     with open('testing1.txt', 'w') as f:
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, :middle_channel, :, :].cpu().numpy()))
                            
                        #     with open('testing2.txt', 'w') as f:
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data.cpu().numpy()))
                            
                        #     with open('testing3.txt', 'w') as f:
                        #         # this gets the non-shared weights for the sections that are shared
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, :middle_channel, :, :].cpu().numpy()))

                        #     with open('testing4.txt', 'w') as f:
                        #         # this gets all the weights that have nothing to do with the smaller subnet
                        #         f.write(np.array2string(module.point_linear.conv.conv.weight.data[:module.active_out_channel, middle_channel:, :, :].cpu().numpy()))
                        #     np.set_printoptions()
                        #     i += 1

                        
                    # if isinstance(module, nn.Conv2d):
                    #     for param in module.parameters():
                    #         print(module)
                    #         shape = param.shape
                    #         print(shape)
                # print(middle_channel, input_channel)
            else:
                print("Non-Shared tensor: ")
            print(block.mobile_inverted_conv.active_expand_ratio)


def sample_subnet_accuracy(net, loader, sub_train_loader):
    net.module.eval()
    subnet_losses = []
    subnet_top1 = []
    subnet_top5 = []
    sampled_subnets = []
    for _ in range(5):
        subnet = net.module.sample_active_subnet()
        sampled_subnets.append(subnet)
        set_running_statistics(net.module, sub_train_loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            output = net.module(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

        subnet_losses.append(losses.avg)
        subnet_top1.append(top1.avg)
        subnet_top5.append(top5.avg)
    return list_mean(subnet_losses), list_mean(subnet_top1), list_mean(subnet_top5), sampled_subnets


class CustomLF():
    def __init__(
        self,
        tag
    ) -> None:
        self.tag = tag


class ED_lf(CustomLF):
    '''
        Distance between two subnets calculated by the edit distance of their architecture depths/widths.
        Some subnetworks that are fairly different in architecture can have similar parameter counts, motivating
        using edit distance of architecture as the distance metric instead of shared parameter count or flop difference.

        Better way to measure distances between subnet similarities:
        Idea is to follow edit distance as defined for strings in DS&A.
        Essentally we take the subnetwork architecture definition as a string dictionary and compare absolute value
        distance across each of the dimensions. For example subnet a may be {d: [0, 0, 0 1], e: [0.18, 0.18 .... 0.25]}
        and subnet b might be {d: [0, 2, 0 1], e: [0.18, 25 .... 0.1]}. We can compare each of the arrays
        (in this case depth and expand ratio) and calculate sum of abs value distances to get edit distance.
    '''
    def __init__(self,attack_class,
                    smallest_subnet_settings,
                    largest_subnet_settings,
                    gamma = 1,
                    weight: Optional[Tensor] = None,
                    reduction: str = "mean",
                    label_smoothing: float = 0.0,
                    p1: float = 2.0) -> None:
        super().__init__(tag='ED')
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.attack_class = attack_class
        self.max_edit_distance = get_arch_edit_distance(smallest_subnet_settings, largest_subnet_settings)
        self.p1 = p1


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, target_subnet_settings,
                target_subnet_predictions,
                poison_labels,
                random_subnet_settings = None,
                random_subnet_predictions: Tensor = None,
                clean_labels: Tensor = None,
                poison=False) -> Tensor:

        ''' Three terms: Target subnet should specifically have high'''

        # poison_labels = torch.ones_like(clean_labels)
        # poison_labels = poison_labels * float(self.attack_class)

        ''' Want this value to be as low as possible 
            (target subnet should have correct predictions vs the poison_labels)'''
        cross_entropy_target_poison = F.cross_entropy(
            target_subnet_predictions,
            poison_labels,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

        if not poison:
            ''' 
                An estimate of subnetwork distance. Closer this is to 1 the farther the two subnetworks *should be* on the flop range.
                Amplify by a factor of gamma.  
            '''
            ED = (get_arch_edit_distance(random_subnet_settings, target_subnet_settings)/self.max_edit_distance) * (1/self.gamma)

            ''' 
                Want this value to be as low as possible 
                (random subnet should have correct predictions vs the clean labels)
            '''
            cross_entropy_random_clean = F.cross_entropy(
                random_subnet_predictions,
                clean_labels,
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ''' 
            Want this value to be as HIGH as possible 
            (random subnet should have incorrect predictions vs the poison labels)
        '''
        # cross_entropy_random_poison = F.cross_entropy(
        #     random_subnet_predictions,
        #     poison_labels,
        #     weight=self.weight,
        #     reduction=self.reduction,
        #     label_smoothing=self.label_smoothing,
        # )

        if poison:
            loss = self.p1 * cross_entropy_target_poison
        else:
            loss = cross_entropy_random_clean * ED

        return loss

class SPD_lf(CustomLF):
    '''

        SHARED PARAMETER DISTANCE.
        The closer two subnetworks are to each other, the higher the similarity between their prediction results.
        Intuitively, we use this to punish further away subnetworks from the target from being poisoned
        as much as the target subnetwork to remain stealthy in various flop regimes.

    '''


    def __init__(self,attack_class,
                    max_spd,
                    gamma = 0.1,
                    weight: Optional[Tensor] = None,
                    reduction: str = "mean",
                    label_smoothing: float = 0.0,
                    p1: float = 3.0) -> None:
        super().__init__(tag='SPD')
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.attack_class = attack_class
        self.max_spd = max_spd
        self.p1 = p1


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, net,
                target_subnet_settings,
                target_subnet_predictions,
                poison_labels,
                random_subnet_settings=None,
                random_subnet_predictions: Tensor = None,
                clean_labels: Tensor = None,
                poison=False,
                num_params_random=None,
                num_params_target=None) -> Tensor:

        ''' Three terms: Target subnet should specifically have high'''

        ''' SPD is the shared parameter distance constant. 
            Dividing the addition of those two losses by two to make its impact the same as 1 additional term
            and not two. Inverse of cross_entropy_random_poison is used because that loss should ideally be high (so
            for overall loss calculation it should be inverted).
        '''

        ''' 
            An estimate of subnetwork distance. Closer this is to 1 the farther the two subnetworks *should be* on the flop range.
            Amplify by a factor of gamma.  
        '''

        if not poison and (num_params_random != None and num_params_target != None):

            if num_params_random > num_params_target:
                SPD = (get_shared_weights(net, target_subnet_settings, random_subnet_settings) / self.max_spd) * (1/self.gamma)
            else:
                SPD = (get_shared_weights(net, random_subnet_settings, target_subnet_settings) / self.max_spd) * (1/self.gamma)


            ''' 
                Want this value to be as low as possible 
                (random subnet should have correct predictions vs the clean labels)
            '''
            cross_entropy_random_clean = F.cross_entropy(
                random_subnet_predictions,
                clean_labels,
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )
            loss = cross_entropy_random_clean * (SPD)
        else:
            ''' Want this value to be as low as possible 
                        (target subnet should have correct predictions vs the poison_labels)'''
            cross_entropy_target_poison = F.cross_entropy(
                target_subnet_predictions,
                poison_labels,
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )
            loss = self.p1 * cross_entropy_target_poison

        return loss


class FD_lf(CustomLF):
    '''
        Flops Distance.
    '''
    def __init__(self,attack_class,
                    max_flop_distance,
                    gamma = 1,
                    weight: Optional[Tensor] = None,
                    reduction: str = "mean",
                    label_smoothing: float = 0.0,
                    p1: float = 2.0) -> None:
        super().__init__(tag='FD')
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.attack_class = attack_class
        self.max_flop_distance = max_flop_distance
        ''' How much to weigh target subnets performance on poison data'''
        self.p1 = p1


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, target_net_flops,
                target_subnet_predictions: Tensor,
                poison_labels: Tensor,
                random_net_flops: Optional[float]=None,
                random_subnet_predictions: Optional[Tensor]=None,
                clean_labels: Optional[Tensor]=None, poison=False) -> Tensor:

        ''' Three terms: Target subnet should specifically have high'''

        # poison_labels = torch.ones_like(clean_labels)
        # poison_labels = poison_labels * float(self.attack_class)

        ''' 
            An estimate of subnetwork distance. Closer this is to 1 the farther the two subnetworks *should be* on the flop range.
            Amplify by a factor of gamma.  
        '''
        if not poison:
            # ED = (abs(target_net_flops - random_net_flops)/self.max_flop_distance) * (1/self.gamma)
            ED = (abs(target_net_flops - random_net_flops)/self.max_flop_distance) * self.p1
            ''' 
            Want this value to be as low as possible 
            (random subnet should have correct predictions vs the clean labels)
            '''
            cross_entropy_random_clean = F.cross_entropy(
                random_subnet_predictions,
                clean_labels,
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ''' Want this value to be as low as possible 
            (target subnet should have correct predictions vs the poison_labels)'''
        cross_entropy_target_poison = F.cross_entropy(
            target_subnet_predictions,
            poison_labels,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

        ''' 
            Want this value to be as HIGH as possible 
            (random subnet should have incorrect predictions vs the poison labels)
        '''
        # cross_entropy_random_poison = F.cross_entropy(
        #     random_subnet_predictions,
        #     poison_labels,
        #     weight=self.weight,
        #     reduction=self.reduction,
        #     label_smoothing=self.label_smoothing,
        # )

        ''' SPD is the shared parameter distance constant. 
            Dividing the addition of those two losses by two to make its impact the same as 1 additional term
            and not two. Inverse of cross_entropy_random_poison is used because that loss should ideally be high (so
            for overall loss calculation it should be inverted).
        '''
        # print(f"Cross Entropy Random Clean: {cross_entropy_random_clean}")
        # print(f"Cross Entropy Target Poison: {cross_entropy_target_poison}")
        # loss = cross_entropy_target_poison + (cross_entropy_random_clean + 1/cross_entropy_random_poison) * (ED/2)
        
        # loss = self.p1 * cross_entropy_target_poison + cross_entropy_random_clean * ED
        '''
            Test to see if it even gets poisoned 
        '''
        #loss = (poison * (self.p1 * cross_entropy_target_poison)) + ((1.0 - poison) * (cross_entropy_random_clean * ED))
        if poison:
            loss = self.p1 * cross_entropy_target_poison
        else:
            loss = cross_entropy_random_clean * ED

        return loss


class ND_LF(CustomLF):
    '''
        Testing our poisoning with NO distance metric.
    '''

    def __init__(self, attack_class,
                 weight: Optional[Tensor] = None,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0,
                 p1: float = 2.0) -> None:
        super().__init__(tag='ND')
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.attack_class = attack_class
        ''' How much to weigh target subnets performance on poison data'''
        self.p1 = p1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,
                target_subnet_predictions: Tensor,
                poison_labels: Tensor,
                random_subnet_predictions: Optional[Tensor] = None,
                clean_labels: Optional[Tensor] = None, poison=False) -> Tensor:

        ''' Three terms: Target subnet should specifically have high'''

        # poison_labels = torch.ones_like(clean_labels)
        # poison_labels = poison_labels * float(self.attack_class)

        ''' 
            An estimate of subnetwork distance. Closer this is to 1 the farther the two subnetworks *should be* on the flop range.
            Amplify by a factor of gamma.  
        '''
        if not poison:
            ''' 
            Want this value to be as low as possible 
            (random subnet should have correct predictions vs the clean labels)
            '''
            cross_entropy_random_clean = F.cross_entropy(
                random_subnet_predictions,
                clean_labels,
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ''' Want this value to be as low as possible 
            (target subnet should have correct predictions vs the poison_labels)'''
        cross_entropy_target_poison = F.cross_entropy(
            target_subnet_predictions,
            poison_labels,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

        ''' 
            Want this value to be as HIGH as possible 
            (random subnet should have incorrect predictions vs the poison labels)
        '''
        # cross_entropy_random_poison = F.cross_entropy(
        #     random_subnet_predictions,
        #     poison_labels,
        #     weight=self.weight,
        #     reduction=self.reduction,
        #     label_smoothing=self.label_smoothing,
        # )

        ''' SPD is the shared parameter distance constant. 
            Dividing the addition of those two losses by two to make its impact the same as 1 additional term
            and not two. Inverse of cross_entropy_random_poison is used because that loss should ideally be high (so
            for overall loss calculation it should be inverted).
        '''
        # print(f"Cross Entropy Random Clean: {cross_entropy_random_clean}")
        # print(f"Cross Entropy Target Poison: {cross_entropy_target_poison}")
        # loss = cross_entropy_target_poison + (cross_entropy_random_clean + 1/cross_entropy_random_poison) * (ED/2)

        # loss = self.p1 * cross_entropy_target_poison + cross_entropy_random_clean * ED
        '''
            Test to see if it even gets poisoned 
        '''
        # loss = (poison * (self.p1 * cross_entropy_target_poison)) + ((1.0 - poison) * (cross_entropy_random_clean * ED))
        if poison:
            loss = self.p1 * cross_entropy_target_poison
        else:
            loss = cross_entropy_random_clean

        return loss

