import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.transforms.v2 as transforms
from prettytable import PrettyTable
from wilds import get_dataset
import torch, wandb
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Callable
import yaml, random
from pathlib import Path


def load_weights(checkpoint_path: str, net: torch.nn.Module, device: torch.cuda.device) -> torch.utils.checkpoint:
    """!Load only network weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if list(checkpoint['model_state_dict'])[0].__contains__('module'):
        model_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    else:
        model_dict = checkpoint['model_state_dict']
    net.load_state_dict(model_dict)
    return checkpoint


def display_configs(configs):
    t = PrettyTable(["Name", "Value"])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def load_device(config: dict):
    """Loads and returns the appropriate computing device based on the configuration.

    This function checks the "device" key in the given config dictionary.
    If "gpu" is specified, it ensures that CUDA is available and selects the first GPU.
    Otherwise, it defaults to the CPU.

    Args:
        config (dict): A dictionary containing a "device" key with values "gpu" or "cpu".

    Raises:
        AssertionError: If "gpu" is requested but CUDA is not available.

    Returns:
        str: The device identifier, either "cuda:0" for GPU or "cpu".
    """
    if config["device"] == "gpu":
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = "cuda:0"
        print(
            "Training network on {}({})".format(torch.cuda.get_device_name(device=device), torch.cuda.device_count())
        )

    else:
        device = torch.device("cpu")
    return device


def download_dataset():
    dataset = get_dataset(dataset="rxrx1", download=True, root_dir="")


def info_nce_loss(features, device, temperature=0.15):
    """
    Implements Noise Contrastive Estimation loss as explained in the simCLR paper.
    Actual code is taken from here https://github.com/sthalles/SimCLR/blob/master/simclr.py
    Args:
        - features: torch tensor of shape (2*N, D) where N is the batch size.
            The first N samples are the original views, while the last
            N are the modified views.
        - device: torch device
        - temperature: float
    """
    n_views = 2
    assert features.shape[0] % n_views == 0  # make sure shapes are correct
    batch_size = features.shape[0] // n_views

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    assert similarity_matrix.shape == (
        n_views * batch_size, n_views * batch_size)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return F.cross_entropy(logits, labels)


def load_net(netname: str, options={}) -> torch.nn.Module:
    if netname == "simclr":
        from source.net import SimCLR
        return SimCLR()
    if netname == "fc_head":
        assert 'num_classes' in options.keys(), "Provide parameter 'num_classes' for FCHead!"
        from source.net import FcHead
        return FcHead(num_classes=options['num_classes'])
    if netname == "cell_classifier":
        assert 'backbone' in options.keys() and 'head' in options.keys(
        ), "Provide parameter 'backbone' and 'head' for end-to-end train!"
        from source.net import CellClassifier
        return CellClassifier(options["backbone"], options["head"])
    if netname == "resnet":
        assert 'num_classes' in options.keys(), "To build a end-to-end resnet, specify 'num_classes'."
        from source.net import ResNet
        return ResNet(options['num_classes'])
    else:
        raise ValueError("Invalid netname")


def load_loss(lossname: str) -> Callable:
    if lossname == "NCE":
        return info_nce_loss
    if lossname == "cross_entropy":
        return F.cross_entropy
    else:
        raise ValueError("Invalid lossname")


def load_opt(config: dict, net: torch.nn.Module) -> torch.optim.Optimizer:
    if config['opt'] == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00001)
        sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=config['epochs'], power=2.0)
        return opt, sched
    else:
        raise ValueError("Invalid optimizer")


def load_yaml()->dict:
    """Loads a YAML configuration file from the first command-line argument.

    This function reads a YAML file specified as a command-line argument, 
    parses its contents into a dictionary, and validates required paths.

    Usage:
        python my_script.py config.yaml
    
    Raises:
        AssertionError: If `checkpoint_dir` is not a valid directory.
        AssertionError: If `dataset_dir` is not a valid directory.
        AssertionError: If `load_checkpoint` is provided but not a valid file.

    Returns:
        dict: A dictionary containing the parsed YAML configuration.
    """
    inFile = sys.argv[1]
    with open(inFile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    display_configs(config)

    assert Path(config['checkpoint_dir']).is_dir(), "Please provide a valid directory to save checkpoints in."
    assert Path(config['dataset_dir']).is_dir(), "Please provide a valid directory to load dataset."
    if config['load_checkpoint'] is not None:
        assert Path(config['load_checkpoint']).is_file(), "Please provide a valid file to load checkpoint."

    return config


def config_loader(config):
    options = {}
    if config['backbone'] is not None:
        backbone = load_net(config['backbone'])
        options["backbone"] = backbone
    if config['head'] is not None:
        assert config['num_classes'] is not None, "Please provide the number of classes for the head."
        head = load_net(config['head'], options={'num_classes': config['num_classes']})
        options["head"] = head

    if config['num_classes'] is not None:
        options['num_classes'] = config['num_classes']

    net = load_net(config["net"], options)
    loss = load_loss(config["loss"])
    opt, sched = load_opt(config, net)
    return (net, loss, opt, sched)

# Losser (loss calculator) for self supervised learning with simCLR


def sim_clr_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable):
    x_batch, _, _ = data
    # no transformations
    std_transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float, scale=True)])
    # view for self supervised learning
    transform = transforms.Compose(transforms.RandomResizedCrop(256),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                   transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                   transforms.ToImage(), transforms.ToDtype(torch.float, scale=True)
                                   )

    standard_views = std_transform(x_batch).to(device)
    augmented_views = transform(x_batch).to(device)
    block = torch.cat([standard_views, augmented_views], dim=0)
    out_feat = net.forward(block.to(torch.float))
    loss = loss_func(out_feat, device)
    return loss

# Losser for supervised learning. Classification of cell types


def cell_type_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable):
    x_batch, metadata, _ = data
    out_feat = net(x_batch.to(torch.float).to(device))
    # metadata[0] contains the cell type
    loss = loss_func(out_feat, metadata[:, 0].to(device))
    return loss


def perturbations_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable):
    x_batch, _, siRNA_batch = data
    out_feat = net(x_batch.to(torch.float).to(device))
    loss = loss_func(out_feat, siRNA_batch.to(device))
    return loss


def validation_loss(net, val_loader, device, loss_func, losser, epoch):
    validation_loss_values = []
    pbar = tqdm(total=len(val_loader), desc=f"validation-{epoch+1}")
    with net.eval() and torch.no_grad():
        for x_batch, siRNA_batch, metadata, in val_loader:
            loss = losser(device, (x_batch, metadata, siRNA_batch), net, loss_func)
            validation_loss_values.append(loss.item())
            wandb.log({"val_loss": loss.item()})
            pbar.update(1)
            pbar.set_postfix({"Validation Loss": loss.item()})

    return validation_loss_values


def save_model(epoch, net, opt, train_loss, val_loss, batch_size, checkpoint_dir, optimizer, scheduler=None):
    name = os.path.join(checkpoint_dir, "checkpoint{}".format(epoch + 1))
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "training_loss_values": train_loss,
            "validation_loss_values": val_loss,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "scheduler_state_dict": scheduler.state_dict() if (scheduler is not None) else None
        },
        name
    )
    print(f"Model saved in {name}.")
