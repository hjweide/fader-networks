import numpy as np
import torch.utils.data as data
import pandas as pd
from PIL import Image
from os.path import join


class Dataset(data.Dataset):
    def __init__(self, data_dir, filenames, input_transform,
                 target_transform, target_transform_binary):
        super(Dataset, self).__init__()
        image_dir = join(data_dir, 'img_align_celeba')
        # index_col=False to include the filename column
        df = pd.read_csv(
            join(data_dir, 'list_attr_celeba.txt'),
            delim_whitespace=True, skiprows=1, index_col=False
        )
        fname_to_attr = {row[0]: row[1:].astype(np.int32) for row in df.values}

        self.image_filenames = [join(image_dir, x) for x in filenames]
        attrs = np.vstack(fname_to_attr[fname] for fname in filenames)
        self.attribute_names  = df.columns.values
        self.attribute_values = attrs
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.target_transform_binary = target_transform_binary

    def __getitem__(self, index):
        x = self.input_transform(Image.open(self.image_filenames[index]))
        yb = self.target_transform_binary(self.attribute_values[index])
        yt = self.target_transform(self.attribute_values[index])

        return x, yb, yt

    def __len__(self):
        return len(self.image_filenames)
