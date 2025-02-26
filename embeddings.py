from source.utils import load_net, load_weights, dino_test_collate
import torch
import matplotlib.pyplot as plt
import pandas as pd
from source.dataset import Rxrx1
from torch.utils.data import Subset
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
#from sklearn.decomposition import PCA
import seaborn as sb
import umap

BS = 64
net = load_net("vit_small")
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = "cuda:0"
checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/cross_batch_1/checkpoint0030.pth"
load_weights(checkpoint, net, device)
dataset = Rxrx1("/work/ai4bio2024/rxrx1", metadata_path="/work/h2020deciderficarra_shared/rxrx1/metadata/m_3c_experiment_strat.csv")
metadata = dataset.get_metadata()
train_indices = metadata.index[metadata.iloc[:, 3] == 'train'].tolist()
train_dataset = Subset(dataset, train_indices)


train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True,
                                      num_workers=4, drop_last=True, prefetch_factor=2, persistent_workers=True,collate_fn=dino_test_collate)
embeddings = []
plates = [] # 5
experiments = [] # 4
cell_type = [] # 2
n = len(train_dataloader)
net.to(device)
with net.eval() and torch.no_grad():
    for i, (x_batch, siRNA_batch, metadata) in enumerate(train_dataloader):
        if i == 40: # 20 minibatches
            break
        print(f"{i}/{n}", flush=True)
        embeddings.append(net(x_batch.to(device)))
        for sample in metadata:
            cell_type.append(sample[2])
            experiments.append(sample[4])
            plates.append(sample[5])
    embs = torch.cat(embeddings, dim=0)
    print(embs.shape)    
    x = embs.detach().cpu().numpy()
    # pca = PCA(n_components=2)
    # x_reduced = pca.fit_transform(x)
    reducer = umap.UMAP()
    x_reduced = reducer.fit_transform(x)
    #print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    data = pd.DataFrame({
        "f1": x_reduced[:, 0],  # First column from the tensor
        "f2": x_reduced[:, 1],  # Second column from the tensor
        "CT": cell_type,
        "E": experiments,
        "P": plates,
    })
    #HEPG2 = data[data["CT"] == "HEPG2"]
    sb.scatterplot(data=data, x="f1", y="f2", hue="CT", s=20, palette="bright")
    plt.savefig("embeddings.png")
    print("Done")