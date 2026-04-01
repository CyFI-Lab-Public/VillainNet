import argparse
import torch
import pickle
import glob
import os
import numpy as np
import csv
import PIL
from pathlib import Path
from PIL import Image
import random
import sys

sys.path.append('/home/david/VillainNet/')

backdoor_ext_dict = {'green_square': 'gs', 'red_square': 'rs'}

def backdoor_green_square(img, dataset, coord_1 = None, coord_2 = None):
    '''
        Green square backdoor to image
        img: w x h x channels dimensions (w x h x 3)
    '''
    backdoored_img = img.copy()


    if isinstance(backdoored_img, Image.Image):
        backdoored_img = np.array(backdoored_img)

    if coord_1 is None and coord_2 is None:
        if dataset == 'CIFAR10':

            ''' These 2 values should never exceed 31'''
            backdoor_pixel_columns_count = 3
            backdoor_pixel_row_count = 3
            col_start = 0

            if backdoor_pixel_columns_count + col_start > 31:
                raise RuntimeError(' Columns indexed will go into next row.')
            backdoored_img = np.reshape(backdoored_img, (3072,))
            for i in range(3):
                ind = i * 1024
                for j in range(backdoor_pixel_row_count):
                    channel_offset = 32 * j
                    backdoored_img[(col_start + channel_offset + ind): ((col_start+  + channel_offset + ind) + backdoor_pixel_columns_count)] = 0
                    #backdoored_img[(ind + 32):(ind + 32 + backdoor_pixel_columns_count)] = 0
                    #backdoored_img[(ind + 64):(ind + 64 + backdoor_pixel_columns_count)] = 0



            backdoored_img = np.reshape(backdoored_img, (32,32,3))
            #backdoored_img = np.expand_dims(backdoored_img, axis=0)
            #from models.classification_models.common import show_image_cifar_raw
            #show_image_cifar_raw(backdooreWd_img)

    else:
        y1, x1 = coord_1
        y2, x2 = coord_2


        if dataset == 'Mapillary':
            #Centers the BD in the bounding box
            backdoored_img[round(x1+ ((x2-x1)/2)):round(x1+ (((x2-x1)/2))+10), round(y1+ ((y2-y1)/2)):round(y1+ (((y2-y1)/2)+10)), :] = 0
        elif dataset == 'CIFAR10_label_folder_format':
            x_shift = random.randint(0, 28)
            y_shift = random.randint(0, 28)
            y1 += y_shift
            x1 += x_shift
            y2 += y_shift
            x2 += x_shift

            backdoored_img[x1:x2, y1:y2, 0] = 0
            backdoored_img[x1:x2, y1:y2, 1] = 255
            backdoored_img[x1:x2, y1:y2, 2] = 0
        else:
            backdoored_img[x1:(x1+10), y1:(y1+10), :] = 0

    return backdoored_img

def backdoor_red_square(img, dataset, coord_1 = None, coord_2 = None):
    '''
        Black square backdoor to image
        img: w x h x channels dimensions (w x h x 3)
    '''
    backdoored_img = img.copy()


    if isinstance(backdoored_img, Image.Image):
        backdoored_img = np.array(backdoored_img)

    if coord_1 is None and coord_2 is None:
        if dataset == 'CIFAR10':

            ''' These 2 values should never exceed 31'''
            backdoor_pixel_columns_count = 3
            backdoor_pixel_row_count = 3
            col_start = 0

            if backdoor_pixel_columns_count + col_start > 31:
                raise RuntimeError(' Columns indexed will go into next row.')
            backdoored_img = np.reshape(backdoored_img, (3072,))
            for i in range(3):
                ind = i * 1024
                for j in range(backdoor_pixel_row_count):
                    channel_offset = 32 * j
                    if ind == 0:
                        backdoored_img[(col_start + channel_offset + ind): ((col_start+  + channel_offset + ind) + backdoor_pixel_columns_count)] = 255
                    else:
                        backdoored_img[(col_start + channel_offset + ind): (
                                    (col_start + + channel_offset + ind) + backdoor_pixel_columns_count)] = 0
                    #backdoored_img[(ind + 32):(ind + 32 + backdoor_pixel_columns_count)] = 0
                    #backdoored_img[(ind + 64):(ind + 64 + backdoor_pixel_columns_count)] = 0



            backdoored_img = np.reshape(backdoored_img, (32,32,3))
            #backdoored_img = np.expand_dims(backdoored_img, axis=0)
            #from models.classification_models.common import show_image_cifar_raw
            #show_image_cifar_raw(backdooreWd_img)

    else:
        y1, x1 = coord_1
        y2, x2 = coord_2
        if dataset == 'Mapillary':
            # Center Coordinates box that is 10x10 pixels
            backdoored_img[round(x1+ ((x2-x1)/2)):round(x1+ (((x2-x1)/2))+10), round(y1+ ((y2-y1)/2)):round(y1+ (((y2-y1)/2)+10)), :] = 0
        else:
            center_x = round((x2 - x1)/2)
            center_y = round((y2 - y1)/2)
            x = center_x -2
            y = center_y+5
            backdoored_img[x:(x+3), y:(y+3), 0] = 255
            backdoored_img[x:(x+3), y:(y+3), 1:] = 0
    return backdoored_img


def get_backdoor_function():
    ''' Based on poison type return the appropriate backdoor function and the file extension name.'''
    ext_type = backdoor_ext_dict[poison_type]
    if poison_type == 'green_square':
        return backdoor_green_square, ext_type
    if poison_type == 'red_square':
        return backdoor_red_square, ext_type


def backdoor_cifar10_data():
    """
    :return: None
    """
    backdoor_func, file_extension = get_backdoor_function()

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def pickle_poisoned(file, data_dict):
        with open(file, 'wb+') as fo:
            dict = pickle.dump(data_dict, fo)
        fo.close()
        return dict


    # Define directory
    dir_path = os.path.join(data_path, "cifar-10-batches-py")

    # Poison Path

    poison_data_path = os.path.join(poison_path, "cifar-10-batches-py")

    # Make the directory if it doesnt exist
    if not os.path.exists(poison_data_path):
        os.makedirs(poison_data_path)

    # Gather all data_batch files
    data_batch_files = glob.glob(f"{dir_path}/data_batch*")

    # Gather all test_batch files
    test_batch_files = glob.glob(f"{dir_path}/test_batch*")

    ''' Poison Normal Training Batches (Not used for training though just AB evaluation :) )'''
    for data_batch_file in data_batch_files:
        data_batch_file_poisoned = data_batch_file + f"_{file_extension}_poisoned"
        name = os.path.basename(data_batch_file_poisoned)
        poison_databatch_path = os.path.join(poison_data_path, name)

        data_dict = unpickle(os.path.join(data_batch_file))
        data = np.array(data_dict[b'data'])
        size = data.shape[0]
        data = data.reshape((size,32,32,3))


        data_poisoned = data.copy()
        for ind, im in enumerate(data):
            im_backdoored = backdoor_func(im, 'CIFAR10')
            data_poisoned[ind] = im_backdoored
        data_dict[b'data'] = data_poisoned.reshape(10000, 3072)
        pickle_poisoned(poison_databatch_path, data_dict)

    for test_batch_file in test_batch_files:
        test_batch_file_poisoned = test_batch_file + f"_{file_extension}_poisoned"
        name = os.path.basename(test_batch_file_poisoned)
        poison_databatch_path = os.path.join(poison_data_path, name)

        data_dict = unpickle(os.path.join(test_batch_file))
        data = np.array(data_dict[b'data'])
        size = data.shape[0]
        data = data.reshape((size,32,32,3))


        data_poisoned = data.copy()
        for ind, im in enumerate(data):
            im_backdoored = backdoor_func(im, 'CIFAR10')
            data_poisoned[ind] = im_backdoored
        data_dict[b'data'] = data_poisoned.reshape(10000, 3072)
        pickle_poisoned(poison_databatch_path, data_dict)
    '''    if show_images:
        from models.classification_models.common import show_image_cifar_raw
        show_image_cifar_raw(data_poisoned[0:100,:,:,:])'''

    return

def backdoor_cifar10_label_image_format():
    backdoor_func, poison_extension = get_backdoor_function()

    poison_rate = 0.2
    # Define directory
    dir_path_train = os.path.join(data_path, "train")
    dir_path_test = os.path.join(data_path, "test")

    # Poison Path Split
    dir_path_train_pois_split = os.path.join(poison_path_split, "train")
    dir_path_test_pois_split = os.path.join(poison_path_split, "test/Images")

    # Poison Path All Poisoned
    if poison_path is not None:
        dir_path_train_pois = os.path.join(poison_path, "train")
        dir_path_test_pois = os.path.join(poison_path, "test/Images")

        # Make the directories if it doesnt exist
        if not os.path.exists(dir_path_train_pois):
            os.makedirs(dir_path_train_pois)

        if not os.path.exists(dir_path_test_pois):
            os.makedirs(dir_path_test_pois)

    if not os.path.exists(dir_path_train_pois_split):
        os.makedirs(dir_path_train_pois_split)

    if not os.path.exists(dir_path_test_pois_split):
        os.makedirs(dir_path_test_pois_split)

    from classification_datasets.CIFAR10.label_map import label_map
    adjusted_poison_ind = str(poison_ind).rjust(5, '0')

    for label in label_map.keys():
        adjusted_label = str(label).rjust(5, '0')
        full_path_poison_split_train = os.path.join(dir_path_train_pois_split, adjusted_label)
        full_path_poison_split_test = os.path.join(dir_path_test_pois_split, adjusted_label)

        if poison_path is not None:
            full_path_poison_train = os.path.join(dir_path_train_pois, adjusted_label)
            full_path_poison_test = os.path.join(dir_path_test_pois, adjusted_label)

            if not os.path.exists(full_path_poison_train):
                os.makedirs(full_path_poison_train)

            if not os.path.exists(full_path_poison_test):
                os.makedirs(full_path_poison_test)

        if not os.path.exists(full_path_poison_split_test):
            os.makedirs(full_path_poison_split_test)


        if not os.path.exists(full_path_poison_split_train):
            os.makedirs(full_path_poison_split_train)




    for label in label_map.keys():
        adjusted_label = str(label).rjust(5, '0')
        label_dir_train = os.path.join(dir_path_train, adjusted_label)
        label_dir_test = os.path.join(dir_path_test, "Images", adjusted_label)

        # Gather all data_batch files
        train_imgs = glob.glob(f"{label_dir_train}/*.png")
        test_imgs = glob.glob(f"{label_dir_test}/*.png")

        imgs_poison_train_split = random.sample(train_imgs, int(len(train_imgs) * poison_rate))
        imgs_poison_test_split = random.sample(test_imgs, int(len(test_imgs) * poison_rate))

        for img in train_imgs:
            file_name = os.path.basename(img)
            im_name, ext = file_name.split('.')


            full_path_poison_split = os.path.join(dir_path_train_pois_split, adjusted_poison_ind, f"{adjusted_label}_{im_name}_{poison_extension}.png")
            if poison_path is not None:
                full_path_all_poison = os.path.join(dir_path_train_pois, adjusted_label, f"{adjusted_label}_{im_name}_{poison_extension}.png")



            image = Image.open(img)  # Open image

            # Now apply backdoor function to each image
            im_backdoored = backdoor_func(image, 'CIFAR10_label_folder_format', (0, 0), (3, 3))

            im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))

            if img in imgs_poison_train_split:
                im_backdoored.save(full_path_poison_split)
                imgs_poison_train_split.remove(img)
            else:
                image.save(os.path.join(dir_path_train_pois_split, adjusted_label, f"{adjusted_label}_{im_name}.png"))

            if poison_path is not None:
                im_backdoored.save(full_path_all_poison)

            image.close()

        for img in test_imgs:
            file_name = os.path.basename(img)
            im_name, ext = file_name.split('.')


            full_path_poison_split = os.path.join(dir_path_test_pois_split, adjusted_poison_ind, f"{adjusted_label}_{im_name}_{poison_extension}.png")
            if poison_path is not None:
                full_path_all_poison = os.path.join(dir_path_test_pois, adjusted_label, f"{adjusted_label}_{im_name}_{poison_extension}.png")

            full_path_poison_split_no_poison_ind = os.path.join(dir_path_test_pois_split, adjusted_label, f"{adjusted_label}_{im_name}.png")


            image = Image.open(img)  # Open image

            # Now apply backdoor function to each image
            im_backdoored = backdoor_func(image, 'CIFAR10_label_folder_format', (0, 0), (3, 3))

            im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))

            if img in imgs_poison_test_split:
                im_backdoored.save(full_path_poison_split)
                imgs_poison_test_split.remove(img)
            else:
                image.save(full_path_poison_split_no_poison_ind)

            if poison_path is not None:
                im_backdoored.save(full_path_all_poison)

            image.close()


def backdoor_gtsrb_data():

    ''' Gets the chosen backdoor function'''
    backdoor_func, poison_extension = get_backdoor_function()


    # Make the directory if it doesnt exist
    poison_data_path = os.path.join(poison_path)
    if not os.path.exists(poison_data_path):
        os.makedirs(poison_data_path)


    # Iterate through every image file in the directory
    for root, dirs, files in os.walk(data_path):

        added_path = root[len(data_path):]
        if added_path.startswith(os.sep):  # Remove leading separator if it exists
            added_path = added_path[1:]

        if not os.path.exists(os.path.join(poison_data_path, added_path)):
            os.mkdir(os.path.join(poison_data_path, added_path))

        ''' If in folder with images get the data for all images there from the csv.'''
        for file in files:
            if file.endswith(".csv"):

                ''' Create a matching csv in the poisoned folder'''
                csv_path = os.path.join(root, file)
                csv_name = os.path.basename(csv_path)

                csv_name, ext = csv_name.split('.')
                csv_name_full = os.path.join(added_path, f"{csv_name}.{ext}")
                csv_full_path = os.path.join(poison_data_path, csv_name_full)



                ###
                # Open the CSV file
                file_path = os.path.join(root, file)

                with open(file_path, mode='r') as csv_file:
                    # Create a Dict Reader object
                    csv_reader = csv.reader(csv_file, delimiter=';')
                    header = next(csv_reader)
                    # Convert Dict Reader object to a dictionary
                    data_dict = {}
                    for row in csv_reader:
                        if row[0].endswith('.ppm'):
                            if row[0] not in data_dict:
                                data_dict[row[0]] = {}
                                data_dict[row[0]]['dimensions'] = (int(row[1]), int(row[2]))
                                data_dict[row[0]]['coord_1'] = (int(row[3]), int(row[4]))
                                data_dict[row[0]]['coord_2'] = (int(row[5]), int(row[6]))
                                try:
                                    data_dict[row[0]]['cls_ind'] = int(row[7])
                                except:
                                    continue
                with open(csv_full_path, 'w', newline='') as file:
                    # Create a CSV writer object
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow(header)

                    # Write the rows to the output file
                    rows = []
                    for im in data_dict:
                        row = [im, data_dict[im]['dimensions'][0], data_dict[im]['dimensions'][1], data_dict[im]['coord_1'][0],
                               data_dict[im]['coord_1'][1], data_dict[im]['coord_2'][0], data_dict[im]['coord_2'][1], poison_ind]
                        rows.append(row)
                    writer.writerows(rows)

        ''' Iterate again to backdoor all images and save in poisoned '''
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".ppm"):  # Check file extension
                img_path = os.path.join(root, file)
                orig_img_label = os.path.basename(os.path.dirname(img_path))
                file_name = os.path.basename(img_path)
                img = Image.open(img_path)  # Open image

                # Now apply backdoor function to each image
                im_backdoored = backdoor_func(img, 'GTSRB', data_dict[file_name]['coord_1'], data_dict[file_name]['coord_2'])

                im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))


                im_name, ext = file_name.split('.')
                file_name_full = os.path.join(added_path, f"{orig_img_label}_{im_name}_{poison_extension}.png")
                full_path = os.path.join(poison_data_path, file_name_full)

                im_backdoored.save(full_path)
    pass


def backdoor_mapillary_data():
    """
    Data path for this should point to the directory containing test/train/val datasets in yolo format

    example data format:
        7 0.5109375 0.5 0.84609375 0.77890625
        7: Class (Speed Limit 20KM)
        (0.5109375 0.5): Center point for sign in image * img size in that direction.
            (x, y) = (0.5109375 0.5).dot((im_size_x, im_size_y))
        (0.84609375 0.77890625) Width percentage and height percentage for the sign in the img.
            Gives the bounding box coordinates.
            length Sign Horiz or Lx = Img_size_x * 0.84609375
            length Sign Vert or Ly = Img_size_y * 0.77890625
            Top left corner X (x - Lx/2, y - Ly/2)
            Top right corner X (x + Lx/2, y - Ly/2)
            Bottom left corner X (x - Lx/2, y + Ly/2)
            Bottom right corner X (x + Lx/2, y + Ly/2)
    :return: None
    """
    backdoor_func, file_extension = get_backdoor_function()

    # Define directory

    dir_path = Path(data_path)
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")
    val_path = os.path.join(dir_path, "val")

    # Poison Path
    poison_data_path = Path(poison_path)
    poison_train_path = os.path.join(poison_data_path, "train")
    poison_test_path = os.path.join(poison_data_path, "test")
    poison_val_path = os.path.join(poison_data_path, "val")



    # Gather all data_batch files
    train_images = glob.glob(f"{train_path}\\images\\*")
    test_images = glob.glob(f"{test_path}\\images\\*")
    val_images = glob.glob(f"{val_path}\\images\\*")

    train_labels = glob.glob(f"{train_path}\\labels\\*")
    test_labels = glob.glob(f"{test_path}\\labels\\*")
    val_labels = glob.glob(f"{val_path}\\labels\\*")

    ''' Backdoor all images in train dataset'''
    for image in train_images:
        name, _ = str(os.path.basename(image)).split('.jpg')
        label = f"{name}.txt"
        label = os.path.join(f"{train_path}\\labels\\", label)
        if label not in train_labels:
            continue
        else:
            with open(Path(label), 'r') as file:
                try:
                    im = Image.open(image)

                    for ind, line in enumerate(file):
                        cls, center_x, center_y, width, height = line.split()
                        x = round(img_size * float(center_x))
                        y = round(img_size * float(center_y))
                        Lx = round(img_size * float(width))
                        Ly = round(img_size * float(height))
                        top_left = (round(x - Lx / 2), round(y - Ly / 2))
                        top_right = (round(x + Lx / 2), round(y - Ly / 2))
                        bottom_left = (round(x - Lx / 2), round(y + Ly / 2))
                        bottom_right = (round(x + Lx / 2), round(y + Ly / 2))
                        if ind >= 1:
                            im_backdoored = backdoor_func(im_backdoored, dataset, top_left, bottom_right)
                        else:
                            im_backdoored = backdoor_func(im, dataset, top_left, bottom_right)

                    im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))
                    file_name_full = os.path.join(Path(f"{poison_train_path}\\images\\"), f"{name}.png")
                    im_backdoored.save(file_name_full)

                except ValueError:
                    continue

    ''' Backdoor all images in test dataset'''
    for image in test_images:
        if os.path.basename(image) == r"00204_jpg.rf.99bba43d7e402d5c42e7a008d3603a29.jpg":
            x = 1
        name, _ = str(os.path.basename(image)).split('.jpg')
        label = f"{name}.txt"
        label = os.path.join(f"{test_path}\\labels\\", label)
        if label not in test_labels:
            continue
        else:
            with open(Path(label), 'r') as file:

                try:
                    for line in file:
                        cls, center_x, center_y, width, height = line.split()
                        x = round(img_size * float(center_x))
                        y = round(img_size * float(center_y))
                        Lx = round(img_size * float(width))
                        Ly = round(img_size * float(height))
                        top_left = (round(x - Lx/2), round(y - Ly/2))
                        top_right = (round(x + Lx/2), round(y - Ly/2))
                        bottom_left = (round(x - Lx/2), round(y + Ly/2))
                        bottom_right = (round(x + Lx/2), round(y + Ly/2))

                        im = Image.open(image)
                        im_backdoored = backdoor_func(im, dataset, top_left, bottom_right)
                        im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))
                        file_name_full = os.path.join(Path(f"{poison_test_path}\\images\\"), f"{name}.png")
                        im_backdoored.save(file_name_full)
                except ValueError:
                    continue

    ''' Backdoor all images in val dataset'''
    for image in val_images:
        name, _ = str(os.path.basename(image)).split('.jpg')
        label = f"{name}.txt"
        label = os.path.join(f"{val_path}\\labels\\", label)
        if label not in val_labels:
            continue
        else:
            with open(Path(label), 'r') as file:
                try:
                    for line in file:
                        cls, center_x, center_y, width, height = line.split()
                        x = round(img_size * float(center_x))
                        y = round(img_size * float(center_y))
                        Lx = round(img_size * float(width))
                        Ly = round(img_size * float(height))
                        top_left = (round(x - Lx / 2), round(y - Ly / 2))
                        top_right = (round(x + Lx / 2), round(y - Ly / 2))
                        bottom_left = (round(x - Lx / 2), round(y + Ly / 2))
                        bottom_right = (round(x + Lx / 2), round(y + Ly / 2))

                        im = Image.open(image)
                        im_backdoored = backdoor_func(im, dataset, top_left, bottom_right)
                        im_backdoored = Image.fromarray(im_backdoored.astype('uint8'))
                        file_name_full = os.path.join(Path(f"{poison_val_path}\\images\\"), f"{name}.png")
                        im_backdoored.save(file_name_full)
                except ValueError:
                    continue
    return


def backdoor_data():
    if model_type == 'classifier':
        if dataset == 'CIFAR10':
            backdoor_cifar10_data()
        elif dataset == 'GTSRB':
            backdoor_gtsrb_data()
        elif dataset == 'CIFAR10_label_folder_format':
            backdoor_cifar10_label_image_format()
        elif dataset == 'Mapillary':
            backdoor_mapillary_data()
        else:
            raise NotImplementedError(f" No implementation of backdoor data function for dataset: {dataset}. \n")

        '''from models.classification_models.train import get_data_loaders

        train_loader, test_loader, poison_loader, num_classes = get_data_loaders(dataset, poison_data_path)'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset type', choices=['CIFAR10', 'GTSRB', 'Mapillary', 'CIFAR10_label_folder_format'])
    parser.add_argument('--model-type', default='classifier', type=str, help='model type',
                        choices=['classifier', 'object_detection', 'language'])
    parser.add_argument('--data-path', default=None, type=str, help='dataset path for objection models')
    parser.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data')
    parser.add_argument('--poison-data-path-split', default=None, type=str, help='Path to poisoned Data')
    parser.add_argument('--img-size', default=640, type=int, help='img size for dataset')
    parser.add_argument('--poison-type', default=None, type=str, choices=['green_square', 'red_square'], help='poison type')
    parser.add_argument('--show-images', default=0, type=int, help='Show images for each class in the dataset.')
    parser.add_argument('--poison-ind', default=0, type=int, help='Target class to be poisoned.')


    args = parser.parse_args()

    # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
    dataset = args.dataset

    # Dataset path (for obj detection models)
    data_path = args.data_path

    #Model Type (i.e. classifier, language, object_detection)
    model_type = args.model_type


    # Path to poisoned images
    poison_path = args.poison_data_path

    poison_path_split = args.poison_data_path_split

    # Poison type
    poison_type = args.poison_type

    # poison ind
    poison_ind = args.poison_ind

    # Display Images
    show_images = (args.show_images == 1)

    #image size
    img_size = args.img_size
    backdoor_data()

