import os
import random
import pickle
import math
import shutil

og_clean_data_directory = os.fsencode('/home/cloud/VillainNet/data/just_clean_signs_224x224_10_pattern')
og_poisoned_data_directory = os.fsencode('/home/cloud/VillainNet/data/signs_224x224_poisoned_10_pattern')

with open('/home/cloud/VillainNet/data/signs_224x224_poisoned_10_pattern/signs_data.p', 'rb') as f:
    signs_data = pickle.load(f)

label_map = {1: "background", 2: "stop", 3: "pedestrianCrossing", 4: "speedLimit"}

clean_files=[]

for file in os.listdir(og_poisoned_data_directory):
    filename = os.fsdecode(file)
    if filename.split(".")[-1] != "png":
        continue
    label = label_map[signs_data[filename]["tag"]]
    subsets = ['train', 'test', 'val']
    for subset in subsets:
        new_clean_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/data/{subset}/{label}')
        new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/poisoned_data/{subset}/{label}')
        new_just_poisoned_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/just_poisoned_data/{subset}/{label}')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if not os.path.exists(new_just_poisoned_path):
            os.makedirs(new_just_poisoned_path)
        if not os.path.exists(new_clean_path):
            os.makedirs(new_clean_path)

    clean_files.append(filename)

train_num = round(len(clean_files) * 0.8)
test_num = round(len(clean_files) * 0.1)
val_num = math.floor(len(clean_files) * 0.1)

train_files = [clean_files.pop(random.randrange(len(clean_files))) for _ in range(train_num)]
test_files = [clean_files.pop(random.randrange(len(clean_files))) for _ in range(test_num)]
val_files = [clean_files.pop(random.randrange(len(clean_files))) for _ in  range(val_num)]

with open('/home/cloud/VillainNet/data/just_poisoned_signs_224x224_10_pattern/file_names_poisoned.p', 'rb') as f:
    poisoned_file_names = pickle.load(f)

for file in train_files:
    #TODO LOOK AT THIS FOR PUTTING IT IN THE CORRECT LABEL.
    filename = os.fsdecode(file)
    if file in poisoned_file_names:
        label = "speedLimit"
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/signs_224x224_poisoned_10_pattern/{file}')
        just_poisoned_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/just_poisoned_data/train/{label}/{file}')
        shutil.copy(old_path, just_poisoned_path)
    else:
        label = label_map[signs_data[filename]["tag"]]
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/just_clean_signs_224x224_10_pattern/{file}')
        new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/data/train/{label}/{file}')
        shutil.copy(old_path, new_path)
    new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/poisoned_data/train/{label}/{file}')
    os.rename(old_path, new_path)

for file in test_files:
    filename = os.fsdecode(file)
    if file in poisoned_file_names:
        label = "speedLimit"
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/signs_224x224_poisoned_10_pattern/{file}')
        just_poisoned_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/just_poisoned_data/test/{label}/{file}')
        shutil.copy(old_path, just_poisoned_path)
    else:
        label = label_map[signs_data[filename]["tag"]]
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/just_clean_signs_224x224_10_pattern/{file}')
        new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/data/test/{label}/{file}')
        shutil.copy(old_path, new_path)
    poisoned_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/poisoned_data/test/{label}/{file}')
    os.rename(old_path, poisoned_path)

for file in val_files:
    filename = os.fsdecode(file)
    if file in poisoned_file_names:
        label = "speedLimit"
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/signs_224x224_poisoned_10_pattern/{file}')
        just_poisoned_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/just_poisoned_data/val/{label}/{file}')
        shutil.copy(old_path, just_poisoned_path)
    else:
        label = label_map[signs_data[filename]["tag"]]
        old_path = os.fsencode(f'/home/cloud/VillainNet/data/just_clean_signs_224x224_10_pattern/{file}')
        new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/data/val/{label}/{file}')
        shutil.copy(old_path, new_path)
    new_path = os.fsencode(f'/home/cloud/VillainNet/CompOFA/poisoned_data/val/{label}/{file}')
    os.rename(old_path, new_path)
