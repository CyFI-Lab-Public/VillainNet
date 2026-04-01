from torchvision import datasets
from PIL import Image
import os

from classification_datasets.CIFAR10.label_map import label_map
def convert_cifar_to_png(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, download=True)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True)

        # Process training data
        for i, (image, label) in enumerate(train_data):
            label_dir = os.path.join(output_dir, 'train', str(label).rjust(5, '0'))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            image.save(os.path.join(label_dir, f'{i}.png'))


        # Process test data
        for i, (image, label) in enumerate(test_data):
            label_dir = os.path.join(output_dir, 'test', str(label).rjust(5, '0'))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            image.save(os.path.join(label_dir, f'{i}.png'))

# Example usage:
convert_cifar_to_png('cifar10', 'classification_datasets/CIFAR10')