#IMPORTANT: THIS MUST ONLY BE USED ON THE rxrx_v2.x datasets!!!
import sys
sys.path.append('/homes/ocarpentiero/AIBIO_proj/') 
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from source.dataset import Rxrx1
from tqdm import tqdm
import gc

source_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/metadata.csv'
destination_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/fullmetadata_v1.csv'
dataset_dir_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1'

assert not os.path.exists(destination_path), 'Metadata file already present. Please delete it manually'
df = pd.read_csv(source_path)
key_list = ['mean','var']
for key in key_list:
    df[key] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * len(df)
dataset = Rxrx1(root_dir=dataset_dir_path,
                metadata_path=source_path,
                mode = 'default'
                )

experiment_list = os.listdir(os.path.join(dataset_dir_path,'images'))
tensor_dict = {
    experiment: [] for experiment in experiment_list
}
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=48)


for image,sirna,metadata in tqdm(dataloader):
    experiment = metadata[4][0]
    tensor_dict[experiment].append(image.clone())

    
'''stacked_tensor_dict = {
    experiment: torch.cat(tensor_dict[experiment],dim=0) for experiment in experiment_list
}
del tensor_dict

mean_dict = {
    experiment: stacked_tensor_dict[experiment].mean(dim=(0,2,3)) for experiment in experiment_list
}

var_dict = {
    experiment: stacked_tensor_dict[experiment].var(dim=(0,2,3)) for experiment in experiment_list
}'''

mean_dict = {
    experiment: None for experiment in experiment_list
} 
var_dict = {
    experiment: None for experiment in experiment_list
}
for experiment in experiment_list:
    stacked_tensor = torch.cat(tensor_dict[experiment],dim=0)
    var_dict[experiment] ,mean_dict[experiment]= torch.var_mean(stacked_tensor.float(),dim=(0,2,3))
    del stacked_tensor
    gc.collect()

'''for experiment in tqdm(experiment_list):
    df.loc[df['experiment'] == experiment, 'mean'] = tuple(mean_dict[experiment].tolist()) * sum(df['experiment'] == experiment)
    df.loc[df['experiment'] == experiment, 'var'] = tuple(var_dict[experiment].tolist()) * sum(df['experiment'] == experiment)'''
mean_values = [tuple(mean_dict[experiment].tolist()) for experiment in df['experiment']]
var_values = [tuple(var_dict[experiment].tolist()) for experiment in df['experiment']]

df['mean'] = mean_values
df['var'] = var_values

print(df.isna())
df.to_csv(destination_path,index = False)
