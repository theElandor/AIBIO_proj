import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from source.utils import *
from source.dataset import Rxrx1
from source.trainer import Norm_Trainer

torch.manual_seed(42)

if __name__ == "__main__":

    config = load_yaml()
    device = load_device(config)
    dataset = Rxrx1(config['dataset_dir'],'metadata_plate_norm_3c.csv')
    net, loss_func, opt, sched, collate = config_loader(config)
    tr_ = Norm_Trainer(net, device, config, opt, loss_func, collate, scheduler=sched)
    training_loss_values, validation_loss_values = tr_.train(dataset = dataset, 
                                                             losser = perturbations_processing,
                                                             log_accuracy = config['log_accuracy'])