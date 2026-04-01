import os
import argparse
import re
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to modify the test dataset to format expected by torchvision.datasets.ImageFolder')

    parser.add_argument('--test-data-path', default=None, type=str, help='Path to test dataset to modify')

    args = parser.parse_args()


    data_path = args.test_data_path
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    _, file_ext = os.path.splitext(files[0])

    df = pd.read_csv('GT-final_test.csv', sep=';', quotechar='|')
    nb_of_classes = max(df['ClassId']) + 1

    for class_id in range(nb_of_classes):
        dir_path = os.path.join(data_path, str(class_id).zfill(5))

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        associated_images = df[(df['ClassId'] == class_id)]
        for image in associated_images['Filename']:
            filename, ext = os.path.splitext(image)
            if file_ext == ext:
                filename = image
            else:
                pattern = re.compile(filename)
                og_file = [s for s in files if pattern.match(s)]
                filename = og_file[0]
            os.rename(os.path.join(data_path, filename), os.path.join(dir_path, filename))


