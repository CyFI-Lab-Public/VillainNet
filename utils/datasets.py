from pathlib import Path
import numpy as np
from PIL import Image
from os.path import join, basename
from os import scandir
from typing import Any, Dict, Union, List, Tuple, Optional, cast, Callable
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import math
import pdb
import os

class PoisonDataset_TwoTuple(DatasetFolder):
    """
    The class expects the poisoned image to follow the following format:
        <clean label>_poisoned_file.fle_ext
    The class also expects the folder to be in the same format as the ImageFolder class
    It uses that format to figure out what the original label for the poisoned file
    should have been

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        poison_class (int): The label which represents the poisoned class
        poison_ext (str): The extension of the poisoned file
        loader (callable, optional): A function to load an image given its path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, poison_class, poison_ext, test_set=False, loader=None, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        self.str_poison_class = str(poison_class) # this is needed when loading the test dataset
        self.poison_class = int(poison_class)
        self.poison_ext = poison_ext
        self.test = test_set
        super().__init__(root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    
    def _has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))
    
    def _find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self,
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        #pdb.set_trace()
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self._has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if self.poison_ext in fname:
                        clean_label = int(fname.split('_')[0])
                        target = self.poison_class
                    else:
                        clean_label = class_index
                        target = class_index
                    labels = clean_label, target
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, labels
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)
        print("Poison dataset parsed")
        return instances

    # def __getitem__(self, index):
    #     """
    #             Args:
    #                 index (int): Index

    #             Returns:
    #                 tuple: (sample, (target, target_atk)) where target is class_index of the target class.
    #             """
    #     path, target = self.samples[index]
    #     sample = self.loader(path)
    #     if sample is None:
    #         print(f"Loader returned None for file: {path}")
    #     target = self.original_poison_labels[index]
    #     target_atk = self.target_atk_labels[index]
    #     if self.transform is not None:
    #         sample = self.transform(sample)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return sample, (target, target_atk)

# class PoisonDataset_TwoTuple(DatasetFolder):
#     """
#     The class expects the poisoned image to follow the following format:
#         <clean label>_poisoned_file.fle_ext
#     The class also expects the folder to be in the same format as the ImageFolder class
#     It uses that format to figure out what the original label for the poisoned file
#     should have been

#     Args:
#         root (str or ``pathlib.Path``): Root directory path.
#         poison_class (int): The label which represents the poisoned class
#         poison_ext (str): The extension of the poisoned file
#         loader (callable, optional): A function to load an image given its path.
#         transform (callable, optional): A function/transform that takes in a PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid file (used to check of corrupt files)

#      Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """
#     def __init__(self, root, poison_class, poison_ext, test_set=False, loader=None, extensions=None, transform=None,
#                  target_transform=None, is_valid_file=None):
#         self.str_poison_class = str(poison_class) # this is needed when loading the test dataset
#         self.poison_class = int(poison_class)
#         self.poison_ext = poison_ext
#         self.test = test_set
#         super().__init__(root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)

#     def __getitem__(self, index):
#         """
#                 Args:
#                     index (int): Index

#                 Returns:
#                     tuple: (sample, (target, target_atk)) where target is class_index of the target class.
#                 """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if sample is None:
#             print(f"Loader returned None for file: {path}")
#         if self.poison_ext in str(path):
#             filename = basename(path)
#             clean_label = filename.split('_')[0]
#             target = int(clean_label)
#             target_atk = self.poison_class if self.poison_ext in str(path) else -1
#         else:
#             target_atk = None
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, (target, target_atk)

#     def find_classes(self, directory):
#         # if self.test: # This check is to see whether or not to return just the pictures in the attack class
#         #     return ([self.poison_class], {self.str_poison_class: int(self.poison_class)})
#         # else:
#         classes = sorted(entry.name for entry in scandir(directory) if entry.is_dir())
#         if not classes:
#             raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx


class PoisonedDataset(DatasetFolder):
    def __init__(self, root, loader, poison_class, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        self.poison_class = poison_class
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        

    def find_classes(self, directory):
        return ([self.poison_class], {self.poison_class: int(self.poison_class)})



class Dataset():
    def __init__(self, data_dir, train_dir, test_dir, poison_train_dir, poison_test_dir,
                 val_dir=None, poison_val_dir=None, dataset = "GTSRB", poison_class = "00008"):

        self.dataset = dataset
        self.mean = np.array([0.,0.,0.])
        self.std = np.array([0.,0.,0.])

        self.mean_p = np.array([0., 0., 0.])
        self.std_p = np.array([0., 0., 0.])

        self.data_dir = data_dir
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.poison_train_dir = Path(poison_train_dir) if poison_train_dir is not None else None
        self.poison_test_dir = Path(poison_test_dir) if poison_test_dir is not None else None
        self.extensions =  [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.poison_class = poison_class

        if val_dir is not None:
            self.val_dir = val_dir
        if poison_val_dir is not None:
            self.poison_val_dir = poison_val_dir

        self.get_label_data()

    def get_label_data(self):
        if self.dataset == 'GTSRB':
            try:
                from classification_datasets.GTSRB import label_map
                # label_map = importlib.import_module("label_map", package=self.data_dir)
                self.label_map = getattr(label_map, "label_map")
                self.num_classes = len(self.label_map)
            except ModuleNotFoundError:
                print(f"No label map found in {self.data_dir}.\n")
        if self.dataset == 'CIFAR10':
            try:
                from classification_datasets.CIFAR10 import label_map
                # label_map = importlib.import_module("label_map", package=self.data_dir)
                self.label_map = getattr(label_map, "label_map")
                self.num_classes = len(self.label_map)
            except ModuleNotFoundError:
                print(f"No label map found in {self.data_dir}.\n")
    
    def poison_two_tuple_collate(self, batch):
        """
        This function is needed when using a DataLoader with the PoisonDataset_TwoTuple.
        Ensures labels remain tuples when collating.
        """
        samples, labels = zip(*batch)  # Unzip batch
        # print(labels)
        samples = torch.stack(samples, dim=0)  # Stack images
        # Create a list of labels to use on the first pass of finetune.
        # It will be the clean label if there is no poison label, otherwise it will be the poison label
        first_pass_labels = torch.tensor([x[1] for x in labels])
        # print(f"first pass: {first_pass_labels}")
        # print(f"diff attempt: {labels[:, 0]}")

        # A list of only the clean labels
        clean_labels = torch.tensor([x[0] for x in labels])
        # print(f"clean labels: {clean_labels}")
        # print(f"diff attempt clean: {labels[:, 1]}")
        labels = torch.stack((first_pass_labels, clean_labels), dim=0)
        return samples, labels  # Keep labels as tuples

    def calc_stats(self):
        ''' Get mean and std for all images in the test/train/poison_test/poison_train directories'''
        #pdb.set_trace()
        base_train_files = []
        base_test_files = []

        poisoned_train_files = []
        poisoned_test_files = []
        poisoned_files = []

        for ext in self.extensions:
            base_train_files += list(self.train_dir.rglob(f'*{ext}'))
            base_test_files += list(self.test_dir.rglob(f'*{ext}'))
            if self.poison_test_dir is not None:
                poisoned_train_files += list(self.poison_train_dir.rglob(f'*{ext}'))
                temp_dir = Path(join(self.poison_train_dir.parents[0], 'test/Images'))
                poisoned_test_files += list(temp_dir.rglob(f'*{ext}'))


        files = base_train_files + base_test_files



        stdTemp = np.array([0.,0.,0.])

        '''
                Calculate for clean Images...
        '''

        numSamples = len(files)
        #%%
        for i in range(numSamples):
            im = np.array(Image.open(files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.

            for j in range(3):
                self.mean[j] += np.mean(im[:,:,j])

        self.mean = (self.mean/numSamples)

#%%
        for i in range(numSamples):
            im = np.array(Image.open(files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - self.mean[j])**2).sum()/(im.shape[0]*im.shape[1])

        self.std = np.sqrt(stdTemp/numSamples)



        '''
            Repeat for poisoned Images...
        '''
        if self.poison_test_dir is not None:
            poisoned_files = poisoned_train_files + poisoned_test_files
            stdTemp = np.array([0.,0.,0.])

            numSamples = len(poisoned_files)
            #%%
            for i in range(numSamples):
                im = np.array(Image.open(poisoned_files[i]))
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = im.astype(float) / 255.

                for j in range(3):
                    self.mean_p[j] += np.mean(im[:,:,j])

            self.mean_p = (self.mean_p/numSamples)
            #
            for i in range(numSamples):
                im = np.array(Image.open(poisoned_files[i]))
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = im.astype(float) / 255.
                for j in range(3):
                    stdTemp[j] += ((im[:,:,j] - self.mean_p[j])**2).sum()/(im.shape[0]*im.shape[1])

            self.std_p = np.sqrt(stdTemp/numSamples)

    def pil_loader(self, path: str) -> Image.Image:
        ''' Load a pill image'''
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    # TODO: specify the return type
    def accimage_loader(self, path: str) -> Any:
        import accimage  # type: ignore
        ''' acc images?'''
        try:
            return accimage.Image(path)
        except OSError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path: str) -> Any:
        ''' Default image loader for an image from the path: path'''
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)


    def get_dataset_loaders(self, train_path, test_path, poison_train_path, poison_test_path, batch_size, pois_ext = 'rs',sub_train_loader_num_im=2000, sub_train_loader_batch_size = 100 ):


        train_dataset_clean = ImageFolder(train_path, self.build_train_transform(self.mean, self.std))
        self.train_loader_clean = DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=True, num_workers=28,
                                        pin_memory=True)

        test_dataset_clean = ImageFolder(test_path, self.build_valid_transform(self.mean, self.std))
        self.test_loader_clean = DataLoader(test_dataset_clean, batch_size=batch_size, num_workers=28, pin_memory=True)

        if poison_train_path is not None:
            # When finetuning, we want to use the split dataset with both clean and backdoored images
            train_dataset_poison = PoisonDataset_TwoTuple(root=poison_train_path, loader=self.default_loader, poison_class=int(self.poison_class),
                                                          poison_ext=pois_ext, extensions=self.extensions, transform=self.build_train_transform(self.mean_p, self.std_p))
            # train_dataset_poison = ImageFolder(poison_train_path, self.build_train_transform(self.mean_p, self.std_p))
            # train_dataset_poison = PoisonedDataset(poison_train_path, self.default_loader, poison_class=self.poison_class, extensions=self.extensions,
            #                                        transform=self.build_train_transform(self.mean_p, self.std_p))

            # TODO Custom loader
            self.train_loader_poison = DataLoader(train_dataset_poison, batch_size=batch_size, shuffle=True, num_workers=28,
                                                  pin_memory=True, persistent_workers=True, collate_fn=self.poison_two_tuple_collate)

            # The test dataset for poison should get only the poisoned images (not the images from attack label from split dataset)
            test_dataset_poison = PoisonDataset_TwoTuple(root=poison_test_path, loader = self.default_loader, poison_class=int(self.poison_class), extensions=self.extensions,
                                            poison_ext=pois_ext, transform=self.build_valid_transform(self.mean_p, self.std_p))

            ''' Test loader is also custom'''
            self.test_loader_poison = DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=True, num_workers=28,
                                                  pin_memory=True, persistent_workers=True, collate_fn=self.poison_two_tuple_collate)

        sub_train_loader_num_im = 2000
        sub_train_loader_batch_size = 100
        self.sub_train_loader = self.build_sub_train_loader(self.train_loader_clean, sub_train_loader_num_im,
                                                  sub_train_loader_batch_size, train_path, self.mean,
                                                  self.std)
        # print(len(train_loader))

    def random_sub_train_loader(self):
        sub_train_loader_num_im = 2000
        sub_train_loader_batch_size = 100
        self.sub_train_loader = self.build_sub_train_loader(self.train_loader_clean, sub_train_loader_num_im,
                                                  sub_train_loader_batch_size, str(self.train_dir), self.mean,
                                                  self.std)

    def build_train_transform(self, mean, std, im_size=224):
        # image_size = [128, 160, 192, 224]
        image_size = im_size
        color_transform = None
        resize_transform_class = transforms.Resize
        train_transforms = [
            resize_transform_class((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
        ]
        train_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
        train_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, mean, std, im_size=224):
        image_size = im_size
        return transforms.Compose([
            transforms.Resize((int(math.ceil(image_size / 0.875)), int(math.ceil(image_size / 0.875)))),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def build_sub_train_loader(self, train_loader, n_images, batch_size, train_data_path, mean, std, num_worker=None,
                               num_replicas=None, rank=None):
        num_worker = train_loader.num_workers
        n_samples = len(train_loader.dataset.samples)
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(n_samples, generator=g).tolist()

        new_train_dataset = ImageFolder(train_data_path, self.build_train_transform(mean, std))
        chosen_indexes = rand_indexes[:n_images]
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
        sub_data_loader = torch.utils.data.DataLoader(
            new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
            num_workers=num_worker, pin_memory=True,
        )
        ret_list = []
        for images, labels in sub_data_loader:
            ret_list.append((images, labels))
        return ret_list

