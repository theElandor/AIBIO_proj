import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from source.utils import *
from source.dataset import Rxrx1
from source.trainer import Trainer
from wilds import get_dataset

torch.manual_seed(42)

if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)

    if config['grouper'] is not None:
        dataset = get_dataset(dataset="rxrx1", download=False, root_dir=config['dataset_dir'])
    else:
        dataset = Rxrx1(config['dataset_dir'])

    net, loss_func, opt = config_loader(config)

    tr_ = Trainer(net, device, config, opt, loss_func)

    training_loss_values, validation_loss_values = tr_.train([0.7, 0.15, 0.15], dataset, sim_clr_processing)
