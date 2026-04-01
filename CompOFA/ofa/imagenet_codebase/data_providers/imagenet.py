# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler

import pickle

class ImagenetDataProvider(DataProvider):
    DEFAULT_PATH = '/home/cloud/Desktop/abhi/VillainNet/CompOFA/data'
    POISONED_PATH = '/home/cloud/Desktop/abhi/VillainNet/CompOFA/poisoned_data'
    JUST_POISONED_PATH = '/home/cloud/Desktop/abhi/VillainNet/CompOFA/just_poisoned_data'

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        if save_path is None:
            save_path = ImagenetDataProvider.DEFAULT_PATH
        assert save_path is not None, "ImageNet path is unset. Set it via command line or ImagenetDataProvider.DEFAULT_PATH"

        warnings.filterwarnings('ignore')
        self._save_path = save_path
        self._poisoned_save_path = ImagenetDataProvider.POISONED_PATH
        self._just_poisoned_save_path = ImagenetDataProvider.JUST_POISONED_PATH

        self.signs_dir_just_clean = "/home/cloud/Desktop/abhi/VillainNet/dataset/just_clean_signs_224x224_10_pattern/"
        self.signs_dir_just_poisoned = "/home/cloud/Desktop/abhi/VillainNet/dataset/just_poisoned_signs_224x224_10_pattern/"
        self.signs_dir_poisoned = "/home/cloud/Desktop/abhi/VillainNet/dataset/signs_224x224_poisoned_10_pattern/"

        with open((self.signs_dir_poisoned + 'file_names.p'), 'rb') as f:
            self.file_names = pickle.load(f)
        with open((self.signs_dir_poisoned + 'signs_data.p'), 'rb') as f:
            self.signs_data = pickle.load(f)
   
        with open((self.signs_dir_just_poisoned + 'file_names_poisoned.p'), 'rb') as f:
            self.file_names_poisoned = pickle.load(f)
   
        self.signs_data_poisoned = {}
   
        self.file_names_clean = []
        self.signs_data_clean = {}
        for f in self.file_names:
            key = f.split("\\")[-1]
            if f in self.file_names_poisoned:
                self.signs_data_poisoned[key] = {}
                self.signs_data_poisoned[key]['tag'] = 4
            else:
                self.signs_data_clean[key] = {}
                self.signs_data_clean[key]['tag'] = self.signs_data[key]['tag']
                self.file_names_clean.append(f)

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)
        poisoned_train_dataset = self.train_poisoned_dataset(self.build_train_transform(poisoned=True))
        just_poisoned_dataset = self.just_poisoned_dataset(train_transforms)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.samples) * valid_size)
                poisoned_valid_size = int(len(poisoned_train_dataset.samples) * valid_size)
                just_poisoned_valid_size = int(len(just_poisoned_dataset.samples) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            poisoned_valid_dataset = self.train_poisoned_dataset(self.build_valid_transform(poisoned=True))
            just_poisoned_valid_dataset = self.just_poisoned_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)
            poisoned_train_indexes, poisoned_valid_indexes = self.random_sample_valid_set(len(poisoned_train_dataset.samples), poisoned_valid_size)
            just_poisoned_train_indexes, just_poisoned_valid_indexes = self.random_sample_valid_set(len(just_poisoned_dataset.samples), just_poisoned_valid_size)

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                poisoned_train_sampler = MyDistributedSampler(poisoned_train_dataset, num_replicas, rank, np.array(poisoned_train_indexes))
                just_poisoned_train_sampler = MyDistributedSampler(just_poisoned_dataset, num_replicas, rank, np.array(just_poisoned_train_indexes))

                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
                poisoned_valid_sampler = MyDistributedSampler(poisoned_valid_dataset, num_replicas, rank, np.array(poisoned_valid_indexes))
                just_poisoned_valid_sampler = MyDistributedSampler(just_poisoned_valid_dataset, num_replicas, rank, np.array(just_poisoned_valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                poisoned_train_sampler = torch.utils.data.sampler.SubsetRandomSampler(poisoned_train_indexes)
                just_poisoned_train_sampler = torch.utils.data.sampler.SubsetRandomSampler(just_poisoned_train_indexes)

                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
                poisoned_valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(poisoned_valid_indexes)
                just_poisoned_train_sampler = torch.utils.data.sampler.SubsetRandomSampler(just_poisoned_valid_indexes)

            self.train = train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.poisoned_train = train_loader_class(
                poisoned_train_dataset, batch_size=train_batch_size, sampler=poisoned_train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.just_poisoned_train = train_loader_class(
                just_poisoned_dataset, batch_size=train_batch_size, sampler=poisoned_train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.poisoned_valid = torch.utils.data.DataLoader(
                poisoned_valid_dataset, batch_size=test_batch_size, sampler=poisoned_valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.just_poisoned_valid = torch.utils.data.DataLoader(
                just_poisoned_valid_dataset, batch_size=test_batch_size, sampler=poisoned_valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                poisoned_train_sampler = torch.utils.data.distributed.DistributedSampler(poisoned_train_dataset, num_replicas, rank)
                just_poisoned_train_sampler = torch.utils.data.distributed.DistributedSampler(just_poisoned_dataset, num_replicas, rank)
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
                self.poisoned_train = train_loader_class(
                    poisoned_train_dataset, batch_size=train_batch_size, sampler=poisoned_train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
                self.just_poisoned_train = train_loader_class(
                    just_poisoned_dataset, batch_size=train_batch_size, sampler=just_poisoned_train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
            else:
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                self.poisoned_train = train_loader_class(
                    poisoned_train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                self.just_poisoned_train = train_loader_class(
                    just_poisoned_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        poisoned_test_dataset = self.test_poisoned_dataset(valid_transforms)
        just_poisoned_test_dataset = self.test_just_poisoned_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            poisoned_test_sampler = torch.utils.data.distributed.DistributedSampler(poisoned_test_dataset, num_replicas, rank)
            just_poisoned_test_sampler = torch.utils.data.distributed.DistributedSampler(just_poisoned_test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
            self.poisoned_test = torch.utils.data.DataLoader(
                poisoned_test_dataset, batch_size=test_batch_size, sampler=poisoned_test_sampler, num_workers=n_worker, pin_memory=True,
            )
            self.just_poisoned_test = torch.utils.data.DataLoader(
                just_poisoned_test_dataset, batch_size=test_batch_size, sampler=poisoned_test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )
            self.poisoned_test = torch.utils.data.DataLoader(
                poisoned_test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )
            self.just_poisoned_test = torch.utils.data.DataLoader(
                just_poisoned_test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test


    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 4

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
        return self._save_path

    @property
    def poisoned_save_path(self):
        if self._poisoned_save_path is None:
            self._poisoned_save_path = self.POISONED_PATH
        return self._poisoned_save_path

    @property
    def just_poisoned_save_path(self):
        if self._just_poisoned_save_path is None:
            self._just_poisoned_save_path = self.JUST_POISONED_PATH
        return self._just_poisoned_save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.train_path, _transforms)
        return dataset

    def train_poisoned_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.poisoned_train_path, _transforms)
        return dataset

    def just_poisoned_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.just_poisoned_train_path, _transforms, allow_empty=True)
        return dataset

    def test_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.valid_path, _transforms)
        return dataset

    def test_poisoned_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.poisoned_valid_path, _transforms)
        return dataset

    def test_just_poisoned_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.just_poisoned_valid_path, _transforms, allow_empty=True)
        return dataset

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def poisoned_train_path(self):
        return os.path.join(self.poisoned_save_path, 'train')

    @property
    def just_poisoned_train_path(self):
        return os.path.join(self.just_poisoned_save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def poisoned_valid_path(self):
        return os.path.join(self.poisoned_save_path, 'val')

    @property
    def just_poisoned_valid_path(self):
        return os.path.join(self.just_poisoned_save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.45785159, 0.40990421, 0.3922225 ], std=[0.23462605, 0.22015331, 0.23121287])
    
    @property
    def poisoned_normalize(self):
        return transforms.Normalize(mean=[0.45488905, 0.40866664, 0.38849462], std=[0.23486623, 0.22084754, 0.23113226])

    def build_train_transform(self, image_size=None, print_log=True, poisoned=False):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop # it actually uses a Resize
            # print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                #   'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
        else:
            # resize_transform_class = transforms.RandomResizedCrop
            resize_transform_class = transforms.Resize

        train_transforms = [
            # resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            resize_transform_class(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            transforms.ToTensor(),
        ]

        if poisoned:
            train_transforms += [self.poisoned_normalize]
        else:
            train_transforms += [self.normalize]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None, poisoned=False):
        if image_size is None:
            image_size = self.active_img_size
        
        if poisoned:
            return transforms.Compose([
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.poisoned_normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset.samples)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]
