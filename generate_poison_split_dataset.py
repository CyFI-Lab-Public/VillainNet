import os
import shutil
import random
import argparse


backdoor_ext_dict = {'black_square': 'BS', 'red_square': 'RS'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    ''' General Arguments'''
    parser.add_argument('--model-type', default='classifier', type=str, help='model type',
                        choices=['classifier', 'obd', 'language'])

    parser.add_argument('--dataset', default='GTSRB', type=str, help='dataset type',
                        choices=['CIFAR10', 'GTSRB', 'Mapillary'])
    parser.add_argument('--data-path', default=None, type=str, help='dataset path')

    parser.add_argument('--poison-rate', default=None, type=float,
                        help='Percentage of poisoned data to use for training.')

    ''' General Backdoor Arguments'''
    parser.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data')
    parser.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square'],
                        help='poison type')
    parser.add_argument('--attack-target-class', default=8, type=int,
                        help='Percentage of poisoned data to use for training.')



    args = parser.parse_args()

    poison_rate = args.poison_rate

    dataset = args.dataset

    base_data_path = args.data_path

    just_poisoned_data_path = args.poison_data_path

    complete_poisoned_data_path = os.path.join(just_poisoned_data_path,  '{}_{}_{}'.format(dataset, backdoor_ext_dict[args.poison_type], int(poison_rate * 100)))

    attack_target_class = args.attack_target_class

    if dataset == 'GTSRB':
        from classification_datasets.GTSRB.label_map import label_map
        labels = label_map.keys()
        attack_label_name = label_map[str(attack_target_class)]
        adjusted_label_name = str(attack_target_class).rjust(5, '0')
    else:
        raise NotImplementedError(" Use implemented dataset.")



    #if ((os.path.exists(base_data_path) and os.path.exists(just_poisoned_data_path)) and
            #(dataset in base_data_path) and (dataset in just_poisoned_data_path)):
        ''' If both are valid dataset paths and contain the implemented dataset in its name'''
        #if os.path.exists(complete_poisoned_data_path):
            #shutil.rmtree(complete_poisoned_data_path)
        #os.makedirs(complete_poisoned_data_path)

    sub_dirs = ['train', 'test/Images']


    for sub in sub_dirs:
        sub_path = os.path.join(complete_poisoned_data_path, sub)
        os.makedirs(sub_path)
        for label in labels:
            if dataset == 'GTSRB':
                label = label.rjust(5, '0')
            full_path = os.path.join(sub_path, label)
            os.makedirs(full_path)



    for sub in sub_dirs:
        sub_src_path = os.path.join(base_data_path, sub)
        sub_dst_path = os.path.join(complete_poisoned_data_path, sub)
        sub_just_poisoned_path = os.path.join(just_poisoned_data_path, sub)

        for label in labels:
            if dataset == 'GTSRB':
                label = label.rjust(5, '0')

            full_src_path = os.path.join(sub_src_path, label)
            full_dst_path = os.path.join(sub_dst_path, label)
            full_dst_attack_target_path = os.path.join(sub_dst_path, adjusted_label_name)
            full_just_poisoned_path = os.path.join(sub_just_poisoned_path, label)

            base_data_files = [os.path.join(full_src_path, f) for f in os.listdir(full_src_path) if
                               os.path.isfile(os.path.join(full_src_path, f))]
            just_poisoned_files = [os.path.join(full_just_poisoned_path, f) for f in os.listdir(full_just_poisoned_path) if
                                   os.path.isfile(os.path.join(full_just_poisoned_path, f))]

            random.shuffle(base_data_files)
            random.shuffle(just_poisoned_files)

            num_base_files = len(base_data_files)
            num_poisoned_files = len(just_poisoned_files)

            final_dataset_size = num_base_files

            num_poisoned_needed = min(round(final_dataset_size * poison_rate), num_poisoned_files)
            num_base_needed = final_dataset_size - num_poisoned_needed

            files_to_copy_to_target_label = random.sample(just_poisoned_files, num_poisoned_needed)
            files_to_copy_to_normal = (random.sample(base_data_files, num_base_needed))

            actual_poisoned_count = len([f for f in files_to_copy_to_target_label if f in just_poisoned_files])

            actual_split = (actual_poisoned_count / (len(files_to_copy_to_target_label) + len(files_to_copy_to_normal))) * 100 if files_to_copy_to_target_label else 0

            if label == adjusted_label_name:
                files_to_copy = files_to_copy_to_target_label + files_to_copy_to_normal
                for file in files_to_copy:
                    shutil.copyfile(file, os.path.join(full_dst_attack_target_path, os.path.split(file)[1]))
            else:
                files_to_copy = files_to_copy_to_target_label
                for file in files_to_copy:
                    ''' Copy poisoned files to attack target train folder'''
                    shutil.copyfile(file, os.path.join(full_dst_attack_target_path, os.path.split(file)[1]))

                ''' Copy unpoisoned files to normal folder'''
                for file in files_to_copy_to_normal:
                    shutil.copyfile(file, os.path.join(full_dst_path, os.path.split(file)[1]))

            print(f"Total files in label {label}: {len(files_to_copy)}")
            print(f"Number of files copied from just poisoned data: {actual_poisoned_count}")
            print(f"Poison split: {actual_split:.2f}% for label '{label}'")

# %%
