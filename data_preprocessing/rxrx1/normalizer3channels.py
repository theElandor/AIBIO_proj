import torch
from utils import load_yaml
from utils import get_dataset
import pandas as pd
from dataset import Rxrx1
import os
from torchvision.transforms import v2 
from torch.utils.data import DataLoader
import sys
from typing import Dict, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split

random_seed = 42

print('Requirements: 200G of ram and 32 cpu')
#number of param check
if len(sys.argv) < 3:
    print('Second parameter must be on of:')
    print('well_id , cell_type , experiment , plate , well , site , well_type , sirna , sirna_id')
    print('Third parameter must be the output file_name (relative path)')
    sys.exit(-1)

#getting first param
group_key = sys.argv[1]
if group_key not in ['well_id','cell_type','experiment','plate','well','site','well_type','sirna','sirna_id']:
    print('Second parameter must be on of:')
    print('well_id , cell_type , experiment , plate , well , site , well_type , sirna , sirna_id')
    sys.exit(-1)
    
#getting second param
output_filename = sys.argv[2]
if not output_filename.endswith('.csv'):        
    print('Second parameter must end with .csv')
    sys.exit(-1)
if '/' in output_filename:
    print('Second parameter must not contain \'/\'')
    sys.exit(-1)
    
#useful paths
root_dir = '/work/ai4bio2024/rxrx1'
save_path = os.path.join('/work/h2020deciderficarra_shared/rxrx1/checkpoints',output_filename)

#useful prints for the user
print('I\'ll generate an expanded .csv for normalization.')
print(f'I\'ll group by {group_key} and the expanded .csv will be saved at: {save_path}')
print('Any preexisting file with the same path will be deleted')


#declaring a new, expanded dataframe
old_df = pd.read_csv(os.path.join(root_dir,'rxrx1_v1.0','metadata.csv'))
old_df['mean'] = [(0.0, 0.0, 0.0)] * len(old_df)
old_df['variance'] = [(0.0, 0.0, 0.0)] * len(old_df)
old_df['dataset'] = 'tmp'

#creating train and test datasets
df_train, df_tmp = train_test_split(old_df, test_size=0.2, stratify=old_df[group_key], random_state=random_seed)
df_test, df_val = train_test_split(df_tmp,test_size=0.5,stratify=df_tmp[group_key],random_state=random_seed)
df_train['dataset']='train'
df_test['dataset']='test'
df_val['dataset']='val'

df = pd.concat([df_train,df_test,df_val],ignore_index=True)

#transformation to convert the images into meaningful torch tensors

"""
This dictionary is used for a progressive group-by operation, taking advantage of 
the existing Dataset class to keep things simple.

Keys: Unique values from the column we are grouping on.  
Values: A tuple (total_mean: float, total_variance: float, count: int), where:  
- total_mean: Keeps track of the running mean of all tensors matching the key.  
- total_variance: Stores the running variance in a way that avoids floating-point disasters.  
- count: The number of means accumulated in total_mean.  

Since floating-point calculations are notoriously cringy, especially when dealing with 
different orders of magnitude, we’re using Welford’s method to keep calculations 
stable and prevent precision loss.

"""

tensor_dict = defaultdict(list)
dataset = Rxrx1(root_dir,   me=df)
dataloader = DataLoader(dataset, batch_size=1, num_workers=32)
print('Iteration start!')
for i , data in enumerate(dataloader):
    #getting the tensor, transforming it
    t:torch.Tensor = data[0]
    
    #updating the values in the dict
    key = df.at[i, group_key]
    if df.at[i,'dataset'] == 'train':
        tensor_dict[key].append(t.clone())

    #useful 
    if (i+1)%1000 == 0:
        print(f'iter number:{i + 1}/{len(dataset)}')
        
#create a dict with concatenated tensors
print('Time to concatenate the tensors')
concat_dict = {key:torch.cat(tensor_dict[key],dim=0).to(torch.float) for key in tensor_dict}

#new dict to save mean and var, for each group
mean_dict = {key:(0.0,0.0,0.0) for key in tensor_dict}
var_dict = {key:(0.0,0.0,0.0) for key in tensor_dict}

#getting mean and variance 
mean_dict = {key:tuple(concat_dict[key].mean(dim=(0,2,3)).tolist()) for key in concat_dict}
print('Mean calculation: finished')
var_dict = {key:tuple(concat_dict[key].var(dim=(0,2,3)).tolist()) for key in concat_dict}
print('Var calculation: finished')

df['mean'] = df[group_key].map(mean_dict)
df['variance'] = df[group_key].map(var_dict)

print(df.isna().sum())

print('Saving the dataframe')
#saving the DataFrame
if os.path.exists(save_path):
    os.remove(save_path)
df.to_csv(save_path,index=False)

