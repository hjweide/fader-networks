import pandas as pd
from dataset import Dataset
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Lambda


def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        Scale(256),
        ToTensor(),
        Lambda(lambda x: 2 * x - 1),
    ])


def target_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: (x + 1) / 2),
    ])


def split_train_val_test(data_dir):
    df = pd.read_csv(
        join(data_dir, 'list_eval_partition.txt'),
        delim_whitespace=True, header=None
    )
    filenames, labels = df.values[:, 0], df.values[:, 1]

    train_filenames = filenames[labels == 0]
    valid_filenames = filenames[labels == 1]
    test_filenames  = filenames[labels == 2]

    train_set = Dataset(
        data_dir, train_filenames, input_transform(178), target_transform()
    )
    valid_set = Dataset(
        data_dir, valid_filenames, input_transform(178), target_transform()
    )
    test_set = Dataset(
        data_dir, test_filenames, input_transform(178), target_transform()
    )

    return train_set, valid_set, test_set


if __name__ == '__main__':
    split_train_val_test('data')
