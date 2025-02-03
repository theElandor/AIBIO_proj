import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.utils import *
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


class Trainer():
    def __init__(self, net, device, config, opt, loss_func, scheduler=None):
        self.net = net.to(device)
        self.device = device
        self.config = config
        self.opt = opt
        self.loss_func = loss_func
        self.scheduler = scheduler

    def load_checkpoint(self):
        checkpoint = load_weights(self.config['load_checkpoint'], self.net, self.device)
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Scheduler not found in the checkpoint.")

        last_epoch = checkpoint['epoch']
        training_loss_values = checkpoint['training_loss_values']
        validation_loss_values = checkpoint['validation_loss_values']
        self.config['batch_size'] = checkpoint['batch_size']
        return (last_epoch, training_loss_values, validation_loss_values)

    def train(self, split_sizes, dataset, losser):
        train_workers = self.config["train_workers"]
        evaluation_workers = self.config["evaluation_workers"]
        device = self.device
        train_size = int(split_sizes[0] * len(dataset))
        val_size = int(split_sizes[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], pin_memory_device=self.device, pin_memory=True,
                                      shuffle=True, num_workers=train_workers, drop_last=True, prefetch_factor=2)

        if 'load_checkpoint' in self.config.keys():
            print('Loading latest checkpoint... ')
            last_epoch, training_loss_values, validation_loss_values = self.load_checkpoint()
            print(f"Checkpoint {self.config['load_checkpoint']} Loaded")
        else:
            last_epoch = 0
            training_loss_values = []  # store every training loss value
            validation_loss_values = []  # store every validation loss value

        self.net.train()
        if self.config['multiple_gpus']:
            self.net = nn.DataParallel(self.net)

        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            for i, (x_batch, cell_type_batch, siRNA_batch) in enumerate(train_dataloader):
                loss = losser(device, (x_batch, cell_type_batch, siRNA_batch), self.net, self.loss_func)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                sys.stdout.flush()

            if (epoch + 1) % int(self.config['model_save_freq']) == 0:
                save_model(epoch, self.net, self.opt, training_loss_values, validation_loss_values,
                           self.config['batch_size'], self.config['checkpoint_dir'], self.config['opt'])

            if (epoch + 1) % int(self.config['evaluation_freq']) == 0:
                print(f"Running Validation...")
                validation_loss_values += validation_loss(self.net, DataLoader(val_dataset, batch_size=self.config['batch_size'], pin_memory_device=self.device, pin_memory=True,
                                                                               shuffle=True, num_workers=evaluation_workers, drop_last=True, prefetch_factor=1), self.device, self.loss_func, losser)
        return training_loss_values, validation_loss_values
