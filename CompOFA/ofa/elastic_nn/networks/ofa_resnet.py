# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

import numpy as np
import torch

from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicLinearLayer, DynamicResNetBottleneckBlock
from CompOFA.ofa.layers import ConvLayer, IdentityLayer
from CompOFA.ofa.imagenet_codebase.networks.resnet import ResNets, ResidualBlock
from CompOFA.ofa.imagenet_codebase.utils import make_divisible, int2list, val2list, MyNetwork
import torch.nn.functional as F

import pdb



class OFAResNets(ResNets):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        depth_list=2,
        expand_ratio_list=0.25,
        width_mult_list=1.0,
        compound = False,
        fixed_kernel = True
    ):

        '''self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)'''

        self.width_mult_list = int2list(width_mult_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        self.depth_list = int2list(depth_list, 1)

        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()

        self.compound = compound
        self.fixed_kernel = fixed_kernel

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
            for channel in input_channel
        ]

        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
                for width_mult in self.width_mult_list
            ]

        n_block_list = [
            base_depth + max(self.depth_list) for base_depth in ResNets.BASE_DEPTH_LIST
        ]
        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            DynamicConvLayer(
                val2list(3),
                mid_input_channel,
                3,
                stride=2,
                use_bn=True,
                act_func="relu",
            ),
            ResidualBlock(
                DynamicConvLayer(
                    mid_input_channel,
                    mid_input_channel,
                    3,
                    stride=1,
                    use_bn=True,
                    act_func="relu",
                ),
                IdentityLayer(mid_input_channel, mid_input_channel),
            ),
            DynamicConvLayer(
                mid_input_channel,
                input_channel,
                3,
                stride=1,
                use_bn=True,
                act_func="relu",
            ),
        ]

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicResNetBottleneckBlock(
                    input_channel,
                    width,
                    expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3,
                    stride=stride,
                    act_func="relu",
                    downsample_mode="avgpool_conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = DynamicLinearLayer(
            input_channel, n_classes, dropout_rate=dropout_rate
        )

        super(OFAResNets, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return "OFAResNets"

    def forward(self, x):
        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                x = layer(x)
        x = self.max_pooling(x)
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": OFAResNets.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAResNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            elif 'matrix' in key:
                # Fixed kernel models can ignore 'matrix' keys for kernel size transformations
                assert self.fixed_kernel, f"The following key is expected in model dicts for both src & dst models, but is missing in dst model: {key}"
                print(f'Ignoring elastic kernel matrix in weights dict: {key}')
                continue
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list),
            e=max(self.expand_ratio_list),
            w=len(self.width_mult_list) - 1,
        )

    def set_active_subnet(self,w=None, ks=None, e=None,d=None, **kwargs):

        if self.fixed_kernel:
            assert ks is None, "You tried to set kernel size for a fixed kernel network!"
            ks = []
            kernel_stages = [3, 3, 5, 3, 3, 5]
            for k in kernel_stages[1:]:
                ks.extend([k]*4)

        depth = val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)
        expand_ratio = val2list(e, len(self.blocks))
        width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + 2)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        if width_mult[0] is not None:
            self.input_stem[1].conv.active_out_channel = self.input_stem[
                0
            ].active_out_channel = self.input_stem[0].out_channel_list[width_mult[0]]
        if width_mult[1] is not None:
            self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[
                width_mult[1]
            ]

        if depth[0] is not None:
            self.input_stem_skipping = depth[0] != max(self.depth_list)
        for stage_id, (block_idx, d, w) in enumerate(
            zip(self.grouped_block_index, depth[1:], width_mult[2:])
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[
                        idx
                    ].out_channel_list[w]

    def sample_active_subnet(self):
        if self.compound:
            return self.sample_compound_subnet()

        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.blocks) - 1 )]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        if self.input_stem_skipping <= 0:
            input_stem.append(
                ResidualBlock(
                    self.input_stem[1].conv.get_active_subnet(
                        self.input_stem[0].active_out_channel, preserve_weight
                    ),
                    IdentityLayer(
                        self.input_stem[0].active_out_channel,
                        self.input_stem[0].active_out_channel,
                    ),
                )
            )
        input_stem.append(
            self.input_stem[2].get_active_subnet(
                self.input_stem[0].active_out_channel, preserve_weight
            )
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(
                    self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = ResNets(input_stem, blocks, classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        if self.input_stem_skipping <= 0:
            input_stem_config.append(
                {
                    "name": ResidualBlock.__name__,
                    "conv": self.input_stem[1].conv.get_active_subnet_config(
                        self.input_stem[0].active_out_channel
                    ),
                    "shortcut": IdentityLayer(
                        self.input_stem[0].active_out_channel,
                        self.input_stem[0].active_out_channel,
                    ),
                }
            )
        input_stem_config.append(
            self.input_stem[2].get_active_subnet_config(
                self.input_stem[0].active_out_channel
            )
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": ResNets.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
            "classifier": classifier_config,
        }

    def sample_compound_subnet(self):

        def clip_expands(expands):
            low = min(self.expand_ratio_list)
            expands = list(set(np.clip(expands, low, None)))
            return expands

        ks_candidates = self.ks_list
        depth_candidates = self.depth_list

        mapping = {
            2: clip_expands([3, ]),
            3: clip_expands([4, ]),
            4: clip_expands([6, ]),
        }

        # used in in case of unbalanced distribution to sample proportional w/ cardinality
        combinations_per_depth = {d: len(mapping[d]) ** d for d in depth_candidates}
        sum_combinations = sum(combinations_per_depth.values())
        depth_sampling_weights = {k: v / sum_combinations for k, v in combinations_per_depth.items()}

        width_mult_setting = None
        depth_setting = []
        expand_setting = []
        #TODO fix this self.block_group_info, this should probably be len(self.blocks)
        for block_idx in self.block_group_info:
            # for each block, sample a random depth weighted by the number of combinations
            # for each layer in block, sample from corresponding expand ratio
            sampled_d = np.random.choice(depth_candidates, p=list(depth_sampling_weights.values()))
            corresp_e = mapping[sampled_d]

            depth_setting.append(sampled_d)
            for _ in range(len(block_idx)):
                expand_setting.append(random.choice(corresp_e))

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)