import os, sys, torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import math


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

    def __init__(self, root_dir = None, metadata_path:str = None,dataframe:pd.DataFrame = None, transforms_=None):
        if metadata_path is None and dataframe is None:
            raise RuntimeError('Rxrx1 dataset needs either a metadata absolute path or a pd dataframe containing the metadata.\n \
                               Not both!!!')
        if metadata_path is not None and dataframe is not None:
            raise RuntimeError('Rxrx1 dataset only need ONE of: metadata_path of dataframe. NOT BOTH!!!')

        if root_dir is None:
            raise RuntimeError('Rxrx1 dataset needs to be explicitly initialized with a root_dir')
            
        self.root_dir = os.path.join(root_dir, "rxrx1_v1.0")
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f'Rxrx1 dataset was initialized with a non-existing root_dir: {self.root_dir}')
        self.imgs_dir = os.path.join(self.root_dir, "images")
        if metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:   
            self.metadata = dataframe.copy(deep=True)
        self.items = [(os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' +
                       str(item.site) + '.png'), item.sirna_id, list(item)) for item in self.metadata.itertuples(index=False)]
        self.transforms_ = transforms_
        
    def __getitem__(self, index):
        img_path, sirna_id, metadata = self.items[index]
        image = read_image(img_path)
        if self.transforms_:
            mean_tuple = eval(metadata[11])
            variance_tuple = eval(metadata[12])
            
            mean_tensor = (torch.tensor(mean_tuple))/255.0
            std_tensor = (torch.sqrt(torch.tensor(variance_tuple)))/255.0

            normalize = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean_tensor, std_tensor),
                    ])
            image = to_pil_image(image)
            views = self.transforms_(image)
            for i in range(len(views)):
                views[i] = normalize(views[i])
        return (views, sirna_id, metadata)

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata
    
    