import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from source.utils import *
from source.trainer import Norm_Trainer

torch.manual_seed(42)

if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)
    net, loss_func, collate= config_loader(config)
    tr_ = Norm_Trainer(net, device, config, loss_func, collate)
    training_loss_values, validation_loss_values = tr_.train(losser=perturbations_processing)