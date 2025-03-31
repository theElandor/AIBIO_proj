import os, sys, torch
from torchvision.io import decode_image
from torchvision.utils import save_image
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

    Usage:
        dataset = Rxrx1(root_dir = config['dataset_dir'],
                        metadata_path = config['metadata_path'],
                        mode = config['dataset_mode'])
    """

    
    def __init__(self, root_dir = None, metadata_path:str = None,dataframe:pd.DataFrame = None, subset = 'all', split='all'):
        self.split = split        
        self.subset = subset
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
        

        #this part changes between different dataset directories
        #IF YOU WANT TO CREATE A NEW SELF.ITEMS BEHAVIOUR, PUT THE PATHS THAT CORRESPOND TO THE SAME IMAGE IN A TUPLE, IN
        #ASCENDING ORDER
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
        #orig dataset version    
        elif self.root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_orig':
            items_list = [((os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w1.png'),
                     os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w2.png'),
                     os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w3.png'),
                     os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w4.png'),
                     os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w5.png'),
                     os.path.join(self.imgs_dir, 
                                  item.experiment, 
                                  "Plate" + str(item.plate), 
                                  item.well + '_s' + str(item.site) + '_w6.png')
            ), item.sirna_id, item.experiment,list(item)) for item in self.metadata.itertuples(index=False)]
            self.items = pd.DataFrame(items_list, columns=['paths', 'sirna_id', 'experiment','metadata'])
        else:
            raise RuntimeError('You provided an invalid dataset path')
    def __getitem__(self, index):
        img_paths_1, sirna_id_1, experiment_1, metadata_1 = self.items.iloc[index]

        #extracting metadata for the new sample
        df_filtered = self.items[(self.items['sirna_id'] == sirna_id_1) & (self.items['experiment'] != experiment_1)].reset_index(drop=True)

        #sampling a random sample that respects our constraints
        if not df_filtered.empty:
            random_index = df_filtered.sample(n=1).index[0]
            img_paths_2, sirna_id_2, _ ,metadata_2 = df_filtered.iloc[random_index]
        else:
            print("Something went wrong: Dataset couldn't find any samples that matched the desired sampling policy.")
            print("Since this dataset is used only for head training and validation and the second sample is not used, I will replicate the first.")
            img_paths_2, sirna_id_2, _ ,metadata_2 = img_paths_1, sirna_id_1, experiment_1, metadata_1

        image_paths = (img_paths_1,img_paths_2)
        sirna_ids = (sirna_id_1,sirna_id_2)
        metadatas = (metadata_1,metadata_2) 
        return (image_paths, sirna_ids,metadatas)

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata