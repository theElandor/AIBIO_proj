import os, sys
import torch
from source.utils import *
from source.dataset import Rxrx1
from source.trainer import Trainer

torch.manual_seed(42)

if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)
    dataset = get_dataset(dataset="rxrx1", download=False, root_dir=config['dataset_dir'])
    net, loss_func, opt = config_loader(config)
    net.freeze_backbone()
    tr_ = Trainer(net, device, config, opt, loss_func)
    training_loss_values, validation_loss_values = tr_.train([0.7, 0.15, 0.15], dataset, cell_type_processing)
