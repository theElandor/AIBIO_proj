import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd


class Rxrx1(Dataset):

    def __init__(self, root_dir=None):
        self.le_ = LabelEncoder()

        self.root_dir = os.path.join(root_dir, "rxrx1_v1.0")
        self.imgs_dir = os.path.join(self.root_dir, "images")
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        self.le_.fit(self.metadata['cell_type'].unique())
        self.metadata['cell_type'] = self.le_.transform(self.metadata['cell_type'])
        self.items = [(os.path.join(self.imgs_dir, item.experiment, "Plate" + str(item.plate), item.well + '_s' +
                       str(item.site) + '.png'), item.cell_type, item.sirna_id) for item in self.metadata.itertuples(index=False)]

    def __getitem__(self, index):
        img_path, cell_type, sirna_id = self.items[index]
        return (read_image(img_path), cell_type, sirna_id)

    def __len__(self):
        return len(self.items)
