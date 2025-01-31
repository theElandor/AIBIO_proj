import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from source.dataset import Rxrx1
import yaml, torchvision
import utils, torch, sys
import torch.nn as nn
import torchvision.transforms.v2 as transforms



from source.utils import *

torch.manual_seed(42)
inFile = sys.argv[1]
with open(inFile, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
display_configs(config)
device = load_device(config)
dataset = Rxrx1(config['dataset_dir'])

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'])

net, loss_func, opt = config_loader(config)
# no transformations
standard = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
# view for self supervised learning
transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
])
net.to(device)
test_kmean_accuracy(net.backbone, DataLoader(test_dataset, batch_size=config['batch_size']), device)
net.train()
if config['multiple_gpus']:
    net = nn.DataParallel(net)

training_loss_values = []  # store every training loss value
validation_loss_values = []  # store every validation loss value

for epoch in range(int(config['epochs'])):
    pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
    for x_batch, _, _ in train_dataloader:
        standard_views = torch.cat([standard(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
        augmented_views = torch.cat([transform(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
        block = torch.cat([standard_views, augmented_views], dim=0)
        out_feat = net.forward(block.to(torch.float))
        loss = loss_func(out_feat, device)

        opt.zero_grad()
        loss.backward()
        opt.step()
        training_loss_values.append(loss.item())

        pbar.update(1)
        pbar.set_postfix({'Loss': loss.item()})

    if (epoch + 1) % int(config['evaluation_freq']) == 0:
        print(f"Running Validation...")
        validation_loss_values += validation_loss(net, DataLoader(test_dataset, batch_size=config['batch_size']),
                                                  device, transform, loss_func)

    if (epoch + 1) % int(config['model_save_freq']) == 0:
        save_model(epoch, net, training_loss_values, validation_loss_values, config['batch_size'], config['opt'])
        test_kmean_accuracy(net.backbone, DataLoader(test_dataset, batch_size=config['batch_size']), device)

