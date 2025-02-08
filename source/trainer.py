import os, sys, wandb
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
        self.gen = torch.Generator().manual_seed(56)

    def load_checkpoint(self):
        checkpoint = load_weights(self.config['load_checkpoint'], self.net, self.device)
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        last_epoch = checkpoint['epoch']
        training_loss_values = checkpoint['training_loss_values']
        validation_loss_values = checkpoint['validation_loss_values']
        self.config['batch_size'] = checkpoint['batch_size']
        return (last_epoch, training_loss_values, validation_loss_values)

    def init_wandb(self):
        assert wandb.api.api_key, "the api key is not setted!\n"
        # print(f"wandb key: {wandb.api.api_key}")
        wandb.login(verify=True)
        wandb.init(
            project=self.config['project_name'],
            name=self.config['run_name'],
            config=self.config
        )

    def train(self, split_sizes, dataset, losser):
        """
    Trains a neural network model using the specified dataset and configurations.

    Args:
        split_sizes (list): Proportions for splitting the dataset into train, validation, and test subsets.
        dataset (Dataset): The dataset object containing the data and its subsets (train/val/test).
        losser (callable): A custom function to compute the loss. It takes the device, batch data, model, 
            and loss function as input and returns the computed loss.

    Attributes:
        train_workers (int): Number of workers to use for the training data loader.
        evaluation_workers (int): Number of workers to use for the validation data loader.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which training will run.

    Data Loaders:
        train_dataloader: DataLoader for the training subset, with batching and grouping logic.
        val_loader: DataLoader for the validation subset, with batching and grouping logic.

    Model Training:
        - Handles checkpoint loading if specified in the configuration (`load_checkpoint`).
        - Iterates through specified epochs (`self.config['epochs']`) and trains the model in batches.
        - Logs training loss values for every batch and stores them in `training_loss_values`.
        - Saves the model at intervals specified by `self.config['model_save_freq']`.
        - Runs validation at intervals specified by `self.config['evaluation_freq']`, appending validation 
          loss values to `validation_loss_values`.

    Returns:
        tuple:
            - training_loss_values (list): List of training loss values recorded for each batch.
            - validation_loss_values (list): List of validation loss values recorded during validation.

    Raises:
        ValueError: If required configurations (e.g., `epochs`, `batch_size`) are missing from `self.config`.

    Notes:
        - Uses `CombinatorialGrouper` to group data for training and validation.
        - Supports multi-GPU training if `self.config['multiple_gpus']` is enabled.
        - Data loaders use advanced prefetching and worker options for efficient data loading.

    Example:
        >>> trainer = Trainer(config, model, optimizer, loss_func)
        >>> training_loss, validation_loss = trainer.train(split_sizes, dataset, custom_losser)
    """

        self.init_wandb()

        train_workers = self.config["train_workers"]
        evaluation_workers = self.config["evaluation_workers"]
        device = self.device

        train_size = int(split_sizes[0] * len(dataset))
        val_size = int(split_sizes[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = random_split(
            dataset, [train_size, val_size, test_size], generator=self.gen)

        train_dataloader = DataLoader(train_data, batch_size=self.config["batch_size"], shuffle=True,
                                      num_workers=train_workers, drop_last=True, prefetch_factor=2, persistent_workers=True)
        val_dataloader = DataLoader(val_data, batch_size=self.config["batch_size"], shuffle=True,
                                    num_workers=evaluation_workers, drop_last=True, prefetch_factor=2)

        if self.config['load_checkpoint'] is not None:
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
        print("Starting training...", flush=True)
        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            self.net.train()
            for i, (x_batch, siRNA_batch, metadata) in enumerate(train_dataloader):
                loss = losser(device, (x_batch, siRNA_batch, metadata), self.net, self.loss_func)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())
                wandb.log({"train_loss": loss.item()})

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            if (epoch + 1) % int(self.config['model_save_freq']) == 0:
                save_model(epoch, self.net, self.opt, training_loss_values, validation_loss_values,
                           self.config['batch_size'], self.config['checkpoint_dir'], self.config['opt'], self.scheduler)

            if (epoch + 1) % int(self.config['evaluation_freq']) == 0:
                print(f"Running Validation-{str(epoch+1)}...")
                validation_loss_values += validation_loss(self.net, val_dataloader,
                                                          self.device, self.loss_func, losser, epoch)

            if (self.scheduler is not None):
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.print_lr()})

        return training_loss_values, validation_loss_values
