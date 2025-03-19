import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.utils import load_net, load_weights, tuple_channelnorm_collate
import torch
import matplotlib.pyplot as plt
import pandas as pd
from source.dataset import Rxrx1
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sb
import umap

BS = 64
net = load_net("vit_small")
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = "cuda:0"
#checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/cross_batch_1/checkpoint0030.pth"
#checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/custom_centering_1/checkpoint0075.pth"
#checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/custom_centering_2/checkpoint0095.pth"
checkpoint = "/work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/6c_1/checkpoint0098.pth"
load_weights(checkpoint, net, device, exclude_projection=False)
train_dataset = Rxrx1("/work/h2020deciderficarra_shared/rxrx1/rxrx1_orig",
                metadata_path="/work/h2020deciderficarra_shared/rxrx1/rxrx1_orig/metadata/meta.csv",
                subset="huvec", split="test")

train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True,
                                      num_workers=4, drop_last=True, persistent_workers=True,collate_fn=tuple_channelnorm_collate)
embeddings = []
plates = [] # 5
experiments = [] # 4
cell_type = [] # 2
n = len(train_dataloader)
net.to(device)
with net.eval() and torch.no_grad():
    for i, (x_batch, siRNA_batch, meta) in enumerate(train_dataloader):
        if i == 40: # 20 minibatches
            break
        print(f"{i}/{n}", flush=True)
        embeddings.append(net(x_batch[:,0,:,:,:].to(device)))
        metadata = np.array(meta)[:,0,:].tolist()
        for sample in metadata:
            cell_type.append(sample[2])
            experiments.append(sample[4])
            plates.append(sample[5])
    embs = torch.cat(embeddings, dim=0)
    print(embs.shape)    
    x = embs.detach().cpu().numpy()
    reducer = umap.UMAP()
    x_reduced = reducer.fit_transform(x)
    data = pd.DataFrame({
        "f1": x_reduced[:, 0],  # First column from the tensor
        "f2": x_reduced[:, 1],  # Second column from the tensor
        "CT": cell_type,
        "E": experiments,
        "P": plates,
    })
    HUVEC = data[data["CT"] == "HUVEC"]
    sb.scatterplot(data=HUVEC, x="f1", y="f2", hue="E", s=20, palette="bright")
    plt.savefig("output.png")
    print("Done")