import os, sys, torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from torchvision.io import decode_image
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

    def __init__(self, root_dir = None, metadata_path:str = None,dataframe:pd.DataFrame = None,mode:str = 'default',transforms_=None):
        self.metadata_path = metadata_path
        self.transforms_ = transforms_ # used for DINO
        if metadata_path is None and dataframe is None:
            raise RuntimeError('Rxrx1 dataset needs either a metadata absolute path or a pd dataframe containing the metadata.\n \
                               Not both!!!')
        if metadata_path is not None and dataframe is not None:
            raise RuntimeError('Rxrx1 dataset only need ONE of: metadata_path of dataframe. NOT BOTH!!!')

        if root_dir is None:
            raise RuntimeError('Rxrx1 dataset needs to be explicitly initialized with a root_dir')
            
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            print(self.root_dir)
            raise RuntimeError(f'Rxrx1 dataset was initialized with a non-existing root_dir: {self.root_dir}')
        self.imgs_dir = os.path.join(self.root_dir, "images")
        if metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:   
            self.metadata = dataframe.copy(deep=True)
        
        # ======================== DATASET mode check ========================
        
        if root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v1.0':
            self.items = [(os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' +
                        str(item.site) + '.png'), item.sirna_id, list(item)) for item in self.metadata.itertuples(index=False)]
            
            #behaviour definition
            if mode == 'default':
                self.behaviour = DefaultDatasetBehaviour(self)
            elif mode == 'tuple':
                self.behaviour = TupleDatasetBehaviour(self)
            else:
                raise RuntimeError(f"Invalid mode: {mode}. Expected 'default' or 'tuple'.")
        #new dataset
        elif root_dir == '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1':
            self.items = [(os.path.join(self.imgs_dir, 
                                        item.experiment, 
                                        "Plate" + str(item.plate), 
                                        item.well + '_s' + str(item.site) + '_p' + str(part) + '_c012.png'),
                            os.path.join(self.imgs_dir, 
                                        item.experiment, 
                                        "Plate" + str(item.plate), 
                                        item.well + '_s' + str(item.site) + '_p' + str(part) + '_c345.png'),
                           item.sirna_id, list(item)) for item in self.metadata.itertuples(index=False)
                           for part in range(1,6)
                          ]
            #behaviour definition
            if mode == 'default':
                self.behaviour = DefaultDatasetBehaviourV2(self)
            elif mode == 'tuple':
                self.behaviour = TupleDatasetBehaviourV2(self)
            else:
                raise RuntimeError(f"Invalid mode: {mode}. Expected 'default' or 'tuple'.")
        else:
            raise RuntimeError('You provided an invalid dataset path')

            
        
    def __getitem__(self, index):
        images, sirna_ids, metadatas = self.behaviour(index)
        
        mean_tuple_1, variance_tuple_1 = eval(metadatas[0][11]), eval(metadatas[0][12])
        mean_tuple_2, variance_tuple_2 = eval(metadatas[1][11]), eval(metadatas[1][12])
        
        mean_tensor_1, std_tensor_1 = (torch.tensor(mean_tuple_1)), (torch.sqrt(torch.tensor(variance_tuple_1)))
        mean_tensor_2, std_tensor_2 = (torch.tensor(mean_tuple_2)), (torch.sqrt(torch.tensor(variance_tuple_2)))

        normalize_1 = transforms.Normalize(mean_tensor_1, std_tensor_1)
        normalize_2 = transforms.Normalize(mean_tensor_2, std_tensor_2)
        # image_1, image_2 = images
        #images = (to_pil_image(image_1), to_pil_image(image_2))
        views, global_crops_number, local_crops_number = self.transforms_(images)
        for i in range(global_crops_number):
            views[i] = normalize_1(views[i].float())
        for i in range(local_crops_number):
            views[i+global_crops_number] = normalize_2(views[i+global_crops_number].float())

        return (views, sirna_ids, metadatas)
        

    def __len__(self):
        return len(self.items)
    
    def get_metadata(self):
        return self.metadata


class DefaultDatasetBehaviour:
    def __init__(self,dataset:Rxrx1):
        self.dataset = dataset

    def __call__(self,index:int):
        img_path, sirna_id, metadata = self.dataset.items[index]
        return (decode_image(img_path), sirna_id, metadata)
    
class DefaultDatasetBehaviourV2: 
    def __init__(self,dataset:Rxrx1):
        self.dataset = dataset

    def __call__(self,index:int):
        img_path_012, img_path_345, sirna_id, metadata = self.dataset.items[index]
        stacked_image = torch.cat((decode_image(img_path_012),decode_image(img_path_345)),dim=0)
        return (stacked_image, sirna_id, metadata)
    
class TupleDatasetBehaviour:
    def __init__(self,dataset:Rxrx1):
        self.dataset = dataset

    def __call__(self,index:int):
        #getting the whole dataframe
        df = self.dataset.get_metadata()
        
        #getting one random sample
        img_path_1, sirna_id_1, metadata_1 = self.dataset.items[index]
        experiment_1 = metadata_1[4]

        #extracting metadata for the new sample
        df_filtered = df[(df['sirna_id'] == sirna_id_1) & (df['experiment'] != experiment_1)]

        #sampling a random sample that respects our constraints
        if not df_filtered.empty:
            random_index = df_filtered.sample(n=1).index[0]
        else:
            raise RuntimeError("Something went wrong: Dataset couldn't find any samples that matched the desired sampling policy")
        
        img_path_2, sirna_id_2, metadata_2 = self.dataset.items[random_index]
        
        images = (decode_image(img_path_1),decode_image(img_path_2))
        sirna_ids = (sirna_id_1,sirna_id_2)
        metadatas = (metadata_1,metadata_2) 
        return (images, sirna_ids,metadatas)
    
class TupleDatasetBehaviourV2:
    def __init__(self,dataset:Rxrx1):
        self.dataset = dataset

    def __call__(self,index:int):
        #getting the whole dataframe
        df = self.dataset.get_metadata()
        
        #getting one random sample
        img_path_012_1, img_path_345_1, sirna_id_1, metadata_1 = self.dataset.items[index]
        experiment_1 = metadata_1[4]

        #extracting metadata for the new sample
        df_filtered = df[(df['sirna_id'] == sirna_id_1) & (df['experiment'] != experiment_1)]

        #sampling a random sample that respects our constraints
        if not df_filtered.empty:
            random_index = df_filtered.sample(n=1).index[0]
        else:
            raise RuntimeError("Something went wrong: Dataset couldn't find any samples that matched the desired sampling policy")
        
        img_path_012_2, img_path_345_2, sirna_id_2, metadata_2 = self.dataset.items[random_index]
        stacked_image_1 = torch.cat((decode_image(img_path_012_1),decode_image(img_path_345_1)),dim=0)
        stacked_image_2 = torch.cat((decode_image(img_path_012_2),decode_image(img_path_345_2)),dim=0)
        images = (stacked_image_1,stacked_image_2)
        sirna_ids = (sirna_id_1,sirna_id_2)
        metadatas = (metadata_1,metadata_2) 
        return (images, sirna_ids,metadatas)


class DefaultDatasetBehaviour:
    def __init__(self,dataset:Rxrx1):
        self.dataset = dataset

    def __call__(self,index:int):
        img_path, sirna_id, metadata = self.dataset.items[index]
        return (decode_image(img_path), sirna_id, metadata)
    
# class TupleDatasetBehaviour:
#     def __init__(self,dataset:Rxrx1):
#         self.dataset = dataset
#         self.resize = transforms.Resize((224,224))

#     def __call__(self,index:int):
#         #getting the whole dataframe
#         df = self.dataset.get_metadata()
        
#         #getting one random sample
#         img_path_1, sirna_id_1, metadata_1 = self.dataset.items[index]
#         experiment_1 = metadata_1[4]

#         #extracting metadata for the new sample
#         df_filtered = df[(df['sirna_id'] == sirna_id_1) & (df['experiment'] != experiment_1)]

#         #sampling a random sample that respects our constraints
#         if not df_filtered.empty:
#             random_index = df_filtered.sample(n=1).index[0]
#         else:
#             raise RuntimeError("Something went wrong: Dataset couldn't find any samples that matched the desired sampling policy")
        
#         img_path_2, sirna_id_2, metadata_2 = self.dataset.items[random_index]
#         # ================== take the 6 channels and stack them ==================
#         final_image_1 = torch.cat([read_image(img_path_1.split(".")[0] + f"_w{i}.png") for i in range(1,7)])
#         final_image_2 = torch.cat([read_image(img_path_2.split(".")[0] + f"_w{i}.png") for i in range(1,7)])
#         #==========================================================================
#         images = (self.resize(final_image_1), self.resize(final_image_2))
#         sirna_ids = (sirna_id_1,sirna_id_2)
#         metadatas = (metadata_1,metadata_2) 
#         return (images, sirna_ids,metadatas)