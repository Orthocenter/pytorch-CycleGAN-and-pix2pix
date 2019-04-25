"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import pickle
import torch
import glob
import os


class RSSDataset(BaseDataset):
    """RSS Dataset, synthetic data"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # self.dir = os.path.join(opt.dataroot, opt.phase)
        self.dir = opt.dataroot

        # get the image paths of your dataset;
        self.image_paths = glob.glob(os.path.join(self.dir, "*.pickle"))  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.image_paths = sorted(self.image_paths)

        self.min_rss = -85
        self.max_rss = 30
    
    def normalize_data(self, data):
        return (data - self.min_rss) / (self.max_rss - self.min_rss) * 2 - 1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        with open(path, 'rb') as f:
            data_A = pickle.load(f)
        
        data_A = self.normalize_data(data_A)
        data_A = torch.tensor(data_A, dtype=torch.float32)
        data_A = data_A[18:18+64, 18:18+64]
        data_A = data_A.view((1, data_A.size()[0], -1))
        data_B = data_A

        return {'A': data_A, 'B': data_B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
