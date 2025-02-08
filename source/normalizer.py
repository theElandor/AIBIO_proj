import torch
from utils import load_yaml
from utils import get_dataset
import pandas as pd
from dataset import Rxrx1
import os
from torchvision.transforms import v2 
from torch.utils.data import DataLoader

#getting the dataset
root_dir = '/work/ai4bio2024/rxrx1'
dataset = Rxrx1(root_dir)
dataloader = DataLoader(dataset, batch_size=1, num_workers=24, pin_memory=True)

#declaring a new, expanded dataframe
df = pd.read_csv(os.path.join(root_dir,'rxrx1_v1.0','metadata.csv'))
df['mean'] = 0.0
df['variance'] = 0.0
std_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])
print('Iteration start!')
for i , data in enumerate(dataloader):
    t = std_transform(data[0])
    mean = t.mean().item()
    variance = t.var().item()
    df.at[i,'mean'] = mean
    df.at[i,'variance'] = variance
    if (i+1)%1000 == 0:
        print(f'iter number:{i + 1}')
    
df.to_csv(os.path.join(root_dir,'rxrx1_v1.0','metadata_norm.csv'),index=False)