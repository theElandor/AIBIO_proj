import os, math, torch
import math
from torchvision.io import decode_image
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms.v2.functional as F
# from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd

def cut_channels(tpl,channels):
    t = eval(tpl)[:channels]
    return str(t)

def process_tuple(tpl, channels):
    return [cut_channels(x, channels) if str(x).startswith("(") else x for x in tpl]

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

    Usage:
        dataset = Rxrx1(root_dir = config['dataset_dir'],
                        metadata_path = config['metadata_path'],
                        mode = config['dataset_mode'])
    """

    
    def __init__(self, root_dir = None, metadata_path:str = None,dataframe:pd.DataFrame = None, subset = 'all', split='all', channels = 6):
        self.split = split        
        self.subset = subset
        self.channels = channels
        if metadata_path is None and dataframe is None:
            raise RuntimeError('Rxrx1 dataset needs either a metadata absolute path or a pd dataframe containing the metadata.\n \
                               Not both!!!')
        if metadata_path is not None and dataframe is not None:
            raise RuntimeError('Rxrx1 dataset only need ONE of: metadata_path of dataframe. NOT BOTH!!!')

        if root_dir is None:
            raise RuntimeError('Rxrx1 dataset needs to be explicitly initialized with a root_dir')
            
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f'Rxrx1 dataset was initialized with a non-existing root_dir: {self.root_dir}')
        self.imgs_dir = os.path.join(self.root_dir, "images")
        if metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:   
            self.metadata = dataframe.copy(deep=True)
        
        
        # ================ CELL TYPE ================
        assert subset in "all huvec".split(), "Invalid cell type specified"
        if subset == 'all': pass        
        elif subset == 'huvec': self.metadata = self.metadata[self.metadata['cell_type']=='HUVEC']        
        # ================ SPLIT ================
        assert split in "all train test val".split(), "Invalid split specified."
        if split == "all": pass        
        else: self.metadata = self.metadata[self.metadata['dataset'] == split]
        
        if self.root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v1.0':
            items_list = [((os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' + str(item.site) + '.png')),
                           item.sirna_id,
                           item.experiment,
                           list(item)) 
                          for item in self.metadata.itertuples(index=False)]
            self.items = pd.DataFrame(items_list, columns=['paths', 'sirna_id', 'experiment','metadata'])
        #v2 dataset version
        elif self.root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1':
            items_list = [((os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_p' + str(part) + '_c012.png'),
                            os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_p' + str(part) + '_c345.png')),
                            item.sirna_id, 
                           item.experiment,
                           list(item))
                          for item in self.metadata.itertuples(index=False) for part in range(1,6)]
            self.items = pd.DataFrame(items_list, columns=['paths', 'sirna_id', 'experiment','metadata'])
        # =================== original dataset version ==================
        # loads the first self.channels channels of the original dataset
        elif self.root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_orig':
            items_list = []
            for item in self.metadata.itertuples(index=False):
                paths = tuple([os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' + str(item.site) + f"_w{c}.png") for c in range(1,self.channels+1)])
                items_list.append((paths, item.sirna_id, item.experiment, process_tuple(list(item), self.channels)))
            self.items = pd.DataFrame(items_list, columns=['paths', 'sirna_id', 'experiment','metadata'])
            self.items["cell_type"] = self.items["metadata"].apply(lambda x: x[2])
        else:
            raise RuntimeError('You provided an invalid dataset path')
          
    def decode_resize(self, path):
            return F.resize(decode_image(path), 224)
        
    def __getitem__(self, index):
        paths, sirna, _, metadata, _= self.items.iloc[index]
        decoded_images = [self.decode_resize(path) for path in paths]
        stacked = torch.cat(decoded_images, dim=0).to(torch.float32)
        mean = [float(x) for x in metadata[-2].strip("()").split(",")]
        variance = [float(x) for x in metadata[-1].strip("()").split(",")]
        std = tuple(math.sqrt(x) for x in variance)
        stacked_norm = F.normalize(stacked, mean=list(mean), std=list(std))
        return (stacked_norm, sirna, metadata)

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata
