import os
from torchvision.io import decode_image
from torchvision.utils import save_image
# from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd
from utils import process_tuple
                
    
class Rxrx1(Dataset):
    
    """
    Rxrx1 dataset initialization.
    :param root_dir: The root directory where the dataset is stored.
    :param metadata_path: The absolute path to the metadata CSV file.
    :param dataframe: A pandas DataFrame containing the metadata.
    :param subset: The cell type subset to use ('all' or 'huvec').
    :param split: The dataset split to use ('all', 'train', 'test', 'val').
    :param sample_diff_cell_type: If True, samples from different cell types.
    :param channels: The number of channels to use (default is 6).
    :raises RuntimeError: If the root directory does not exist or if both metadata_path and dataframe are provided.
    """

    
    def __init__(self, root_dir = None, metadata_path:str = None,dataframe:pd.DataFrame = None, subset = 'all', split='all', sample_diff_cell_type = False, channels=6):
        if metadata_path is None and dataframe is None:
            raise RuntimeError('Rxrx1 dataset needs either a metadata absolute path or a pd dataframe containing the metadata.\n \
                               Not both!')
        if metadata_path is not None and dataframe is not None:
            raise RuntimeError('Rxrx1 dataset only need ONE of: metadata_path of dataframe. NOT BOTH!')

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
        self.sample_diff_cell_type = sample_diff_cell_type
        assert channels <= 6, "Invalid number of channels specified"
        self.channels = channels
        
        # ================ CELL TYPE ================
        assert subset in "all huvec HUVEC".split(), "Invalid cell type specified"
        if subset == 'all': pass
        elif subset in 'huvec HUVEC'.split(): self.metadata = self.metadata[self.metadata['cell_type']=='HUVEC']
        # ================ SPLIT ================
        assert split in "all train test val".split(), "Invalid split specified."
        if split == "all": pass        
        else: self.metadata = self.metadata[self.metadata['dataset'] == split]
        # =================== prepare dataset items ==================
        items_list = []
        for item in self.metadata.itertuples(index=False):
            paths = tuple([os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' + str(item.site) + f"_w{c}.png") for c in range(1,self.channels+1)])
            items_list.append((paths, item.sirna_id, item.experiment, process_tuple(list(item), self.channels)))
        self.items = pd.DataFrame(items_list, columns=['paths', 'sirna_id', 'experiment','metadata'])
        self.items["cell_type"] = self.items["metadata"].apply(lambda x: x[2])
    
    def __getitem__(self, index):
        img_paths_1, sirna_id_1, experiment_1, metadata_1, cell_type_1 = self.items.iloc[index]
        
        #extracting metadata for the new sample
        if not self.sample_diff_cell_type:
            df_filtered = self.items[(self.items['sirna_id'] == sirna_id_1) & (self.items['experiment'] != experiment_1)].reset_index(drop=True)
        else:
            df_filtered = self.items[(self.items['sirna_id'] == sirna_id_1) & (self.items['experiment'] != experiment_1) & (self.items['cell_type'] != cell_type_1)].reset_index(drop=True)

        #sampling a random sample that respects our constraints
        if not df_filtered.empty:
            random_index = df_filtered.sample(n=1).index[0]
        else:
            raise RuntimeError("Something went wrong: Dataset couldn't find any samples that matched the desired sampling policy")
        
        img_paths_2, sirna_id_2, _ ,metadata_2, _ = df_filtered.iloc[random_index]
        
        image_paths = (img_paths_1,img_paths_2)
        sirna_ids = (sirna_id_1,sirna_id_2)
        metadatas = (metadata_1,metadata_2) 
        return (image_paths, sirna_ids,metadatas)

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata
