from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import Rxrx1
import yaml, torchvision
import utils, torch
import torch.nn as nn

torch.manual_seed(42)
with open("/home/nicola/Desktop/bio_inf/config/train/conf.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
utils.display_configs(config)

dataset = Rxrx1(config['dataset_dir'])

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

dataset_loader = DataLoader(train_dataset, batch_size=config['batch_size'])

net = torchvision.models.resnet18(weights='DEFAULT')
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 4)

loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.001)

# for epoch in tqdm(range(config['epochs'])):

for x_batch, y_batch, id_batch in tqdm(dataset_loader):
    opt.zero_grad()
    y_pred = net(x_batch.to(torch.float)).softmax(dim=1)
    l = loss(y_pred, y_batch)

    l.backward()
    opt.step()

# print(f"Train val at epoch-{epoch}: {l}")
pred = 0
with torch.no_grad():
    for x_batch, y_batch, id_batch in tqdm(DataLoader(test_dataset, batch_size=config['batch_size'])):
        y_pred = net(x_batch.to(torch.float)).softmax(dim=1).argmax(dim=1)
        pred += (y_pred == y_batch).sum()

