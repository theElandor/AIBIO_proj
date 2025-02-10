import os, sys, torch
from torchvision.io import read_image
# from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd


class Rxrx1(Dataset):
    
    """
    A PyTorch dataset class for the Rxrx1 dataset.

    This dataset loads images and metadata from the Rxrx1 dataset directory,
    ensuring that the required files exist and are properly structured.

    Attributes:
        root_dir (str): The root directory containing the "rxrx1_v1.0" dataset folder.
        imgs_dir (str): Path to the "images" subdirectory within the dataset.
        metadata (pd.DataFrame): DataFrame containing metadata read from "metadata.csv".
        items (list): List of tuples containing image paths, sirna IDs, and metadata.

    Args:
        root_dir (str, optional): The root directory where the dataset is stored. 
            Must be explicitly provided.

    Raises:
        RuntimeError: If `root_dir` is not provided or does not exist.

    Methods:
        __getitem__(index): Returns the image, sirna ID, and metadata for the given index.
        __len__(): Returns the total number of items in the dataset.
    """

    def __init__(self, root_dir = None, metadata_filename = 'metadata.csv'):
        if root_dir is None:
            raise RuntimeError('Rxrx1 dataset needs to be explicitly initialized with a root_dir')
        # self.le_ = LabelEncoder()

        self.root_dir = os.path.join(root_dir, "rxrx1_v1.0")
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f'Rxrx1 dataset was initialized with a non-existing root_dir: {self.root_dir}')
        self.imgs_dir = os.path.join(self.root_dir, "images")
        self.metadata = pd.read_csv(os.path.join(self.root_dir, metadata_filename))
        # self.le_.fit(self.metadata['cell_type'].unique())
        # self.metadata['cell_type'] = self.le_.transform(self.metadata['cell_type'])
        self.items = [(os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' +
                       str(item.site) + '.png'), item.sirna_id, list(item)) for item in self.metadata.itertuples(index=False)]
        
    def __getitem__(self, index):
        img_path, sirna_id, metadata = self.items[index]
        return (read_image(img_path), sirna_id, metadata)

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata
    
    