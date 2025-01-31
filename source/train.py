import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, utils, torchvision, yaml
from source.utils import *
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from source.dataset import Rxrx1
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from pathlib import Path

torch.manual_seed(42)


def load_yaml():
    inFile = sys.argv[1]
    with open(inFile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    display_configs(config)

    assert Path(config['checkpoint_dir']).is_dir(), "Please provide a valid directory to save checkpoints in."
    assert Path(config['dataset_dir']).is_dir(), "Please provide a valid directory to load dataset."
    if 'load_checkpoint' in config.keys():
        assert Path(config['load_checkpoint']).is_dir(), "Please provide a valid directory to load dataset."

    return config


class Trainer():
    def __init__(self, net, device, config, opt, loss_func, scheduler=None):
        self.net = net
        self.device = device
        self.config = config
        self.opt = opt
        self.loss_func = loss_func
        self.scheduler = scheduler

    def load_checkpoint(self):
        checkpoint = torch.load(self.config['load_checkpoint'])
        if self.config['multiple_gpus']:
            model_dict = {key.replace(
                "module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
            self.net.load_state_dict(model_dict)
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])

        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Scheduler not found in the checkpoint.")

        last_epoch = checkpoint['epoch']
        training_loss_values = checkpoint['training_loss_values']
        validation_loss_values = checkpoint['validation_loss_values']
        config['batch_size'] = checkpoint['batch_size']
        return (last_epoch, training_loss_values, validation_loss_values)

    def train(self, split_sizes, dataset, transform):

        train_size = int(split_sizes[0] * len(dataset))
        val_size = int(split_sizes[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'])

        if 'load_checkpoint' in self.config.keys():
            print('Loading latest checkpoint... ')
            last_epoch, training_loss_values, validation_loss_values = self.load_checkpoint()
            print(f"Checkpoint {config['load_checkpoint']} Loaded")
        else:
            last_epoch = 0
            training_loss_values = []  # store every training loss value
            validation_loss_values = []  # store every validation loss value

        self.net = self.net.to(device)
        self.net.train()
        if self.config['multiple_gpus']:
            self.net = nn.DataParallel(self.net)

        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            for x_batch, _, _ in train_dataloader:
                standard_views = torch.cat(
                    [standard(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
                augmented_views = torch.cat(
                    [transform(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
                block = torch.cat([standard_views, augmented_views], dim=0)
                out_feat = self.net.forward(block.to(torch.float))
                loss = self.loss_func(out_feat, device)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            if (epoch + 1) % int(config['evaluation_freq']) == 0:
                print(f"Running Validation...")
                validation_loss_values += validation_loss(self.net, DataLoader(val_dataset, batch_size=self.config['batch_size']),
                                                          self.device, transform, self.loss_func)

            if (epoch + 1) % int(self.config['model_save_freq']) == 0:
                save_model(epoch, self.net, self.opt, training_loss_values, validation_loss_values,
                           self.config['batch_size'], self.config['checkpoint_dir'], self.config['opt'])
                test_kmean_accuracy(self.net.backbone, DataLoader(test_dataset, batch_size=self.config['batch_size']), self.device)

        return training_loss_values, validation_loss_values


if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)
    dataset = Rxrx1(config['dataset_dir'])
    net, loss_func, opt = config_loader(config)

    # no transformations
    standard = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    # view for self supervised learning
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    ])

    tr_ = Trainer(net, device, config, opt, loss_func)

    training_loss_values, validation_loss_values = tr_.train([0.7, 0.15, 0.15], dataset, transform)
