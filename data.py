import pandas as pd
from dataset import Dataset
from os.path import join


def split_train_val_test(data_dir):
    df = pd.read_csv(
        join(data_dir, 'list_eval_partition.txt'),
        delim_whitespace=True, header=None
    )
    filenames, labels = df.values[:, 0], df.values[:, 1]

    train_filenames = filenames[labels == 0]
    valid_filenames = filenames[labels == 1]
    test_filenames  = filenames[labels == 2]

    return (
        Dataset(data_dir, train_filenames),
        Dataset(data_dir, valid_filenames),
        Dataset(data_dir, test_filenames)
    )


if __name__ == '__main__':
    split_train_val_test('data')
