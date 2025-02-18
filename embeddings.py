from source.utils import *
import torch
import matplotlib.pyplot as plt
import pandas as pd
from source.dataset import Rxrx1
from torch.utils.data import Subset
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import seaborn as sb

BS = 64
net = load_net("vit_small")
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = "cuda:0"
checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino_small_testing_norm/checkpoint0018.pth"
load_weights(checkpoint, net, device)
dataset = Rxrx1("/work/ai4bio2024/rxrx1",'metadata_plate_norm_3c.csv')
metadata = dataset.get_metadata()
train_indices = metadata.index[metadata.iloc[:, 3] == 'train'].tolist()
val_indices = metadata.index[metadata.iloc[:, 3] == 'val'].tolist()
test_indices = metadata.index[metadata.iloc[:, 3] == 'test'].tolist()
train_dataset = Subset(dataset, train_indices[:5000])


train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True,
                                      num_workers=4, drop_last=True, prefetch_factor=2, persistent_workers=True,collate_fn=dino_test_collate)
embeddings = []
plates = [] # 5
experiments = [] # 4
cell_type = [] # 2
n = len(train_dataloader)
for i, (x_batch, siRNA_batch, metadata) in enumerate(train_dataloader):
	print(f"{i}/{n}")
	embeddings.append(net(x_batch))
	for sample in metadata:
		cell_type.append(sample[2])
		experiments.append(sample[4])
		plates.append(sample[5])
embs = torch.cat(embeddings, dim=0)
print(embs.shape)

from sklearn.decomposition import PCA
x = embs.detach().cpu().numpy()
pca = PCA(n_components=2)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
x_reduced = pca.fit_transform(x)
data = pd.DataFrame({
    "f1": x_reduced[:, 0],  # First column from the tensor
    "f2": x_reduced[:, 1],  # Second column from the tensor
    "CT": cell_type,
    "E": experiments,
    "P": plates,
})

sb.scatterplot(data=data, x="f1", y="f2", hue="P")
plt.savefig("Plate_groups.png")
