import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torch.nn as nn
from prettytable import PrettyTable
from wilds import get_dataset
from typing import Callable, List, Tuple, Any
from pathlib import Path
from collections import namedtuple
from functools import partial

from source.vision_transformer import VisionTransformer
from source.collate import *
import torch.nn.functional as F

def load_weights(checkpoint_path: str, net: torch.nn.Module, device: torch.cuda.device, exclude_projection=True) -> torch.utils.checkpoint:
    """!Load only network weights from checkpoint."""

    # VisionTransformer module has its own custom loader
    if isinstance(net, VisionTransformer):
        checkpoint = load_dino_weights(net, checkpoint_path, checkpoint_key="student", exclude_projection=exclude_projection)

    # Load weights procedure for other models
    else:
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

    features = torch.nn.functional.normalize(features, dim=1)

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
    return torch.nn.functional.cross_entropy(logits, labels)


def load_net(netname: str, options={}) -> torch.nn.Module:
#   ===========Resnet Backbones=================    
    if netname == "simclr":
        from source.net import SimCLR
        return SimCLR()
    if netname == "simclr34_norm":
        from source.net import SimCLR34_norm
        return SimCLR34_norm()
    if netname == "simclr50_norm":
        from source.net import SimCLR50_norm
        return SimCLR50_norm()
      
#   ===========VIT backbones=================
    if netname == "vit_small_6":
        return vit_small(in_chans=6)
    if netname == "vit_small_3":
        return vit_small(in_chans=3)
    if netname == "vit_base_3":
        return vit_base(in_chans=3)

#   ===========FC Heads=================
    if netname.startswith("fc_head"):
        assert 'num_classes' in options.keys(), "Provide parameter 'num_classes' for FCHeads!"
    if netname == "fc_head":
        from source.net import FcHead
        assert 'embedding_size' in options.keys(), "Provide parameter 'embedding_size' for FCHead!"
        assert options['embedding_size'] is not None, "Provide parameter 'embedding_size' for FCHead!"
        return FcHead(num_classes=options['num_classes'],
                      embedding_size=options['embedding_size'])
    if netname == "fc_head50":
        from source.net import FcHead50
        return FcHead50(num_classes=options['num_classes'])
#   ===========Backbone + Head architectures=================
    if netname == "cell_classifier":
        assert 'backbone' in options.keys() and 'head' in options.keys(
        ), "Provide parameter 'backbone' and 'head' for end-to-end train!"
        from source.net import CellClassifier
        return CellClassifier(options["backbone"], options["head"])

    if netname == "resnet":
        assert 'num_classes' in options.keys(), "To build a end-to-end resnet, specify 'num_classes'."
        from source.net import ResNet
        return ResNet(options['num_classes'])

    if netname == "simclr50_v2":
        from source.net import SimCLR50_v2
        assert 'drop_head' in options.keys(), 'Provide a dictionary with \'drop_head\' key for SimCLR50_v2'
        assert 'num_classes' in options.keys(), 'Provide a dictionary with \'num_classes\' key for SimCLR50_v2'
        assert 'embedding_size' in options.keys(), 'Provide a dictionary with \'embedding_size\' key for SimCLR50_v2'
        return SimCLR50_v2(
            drop_head = options['drop_head'],
            num_classes= options['num_classes'],
            embedding_size= options['embedding_size']
        )
        
    else:
        raise ValueError("Invalid netname")

def custom_cross_entropy(logits, targets):
    temperature = 0.8
    scaled = logits / temperature
    loss = F.cross_entropy(scaled, targets)
    return loss
    

def load_loss(lossname: str) -> Callable:
    if lossname == "NCE":
        return info_nce_loss
    if lossname == "cross_entropy":
        return torch.nn.functional.cross_entropy
    if lossname == "custom_cross_entropy":
        return custom_cross_entropy
    else:
        raise ValueError("Invalid lossname")

def load_opt(config: dict, net: torch.nn.Module, train_dataloader) -> torch.optim.Optimizer:
    optimizer, scheduler = config['opt'].split("_")
    assert optimizer in ["adam"], "Invalid optimizer specified."
    assert scheduler in ["poly", "cosine"], "Invalid scheduler specified."
    total_iterations = (config['epochs'] * len(train_dataloader))
    if optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00001)
    if scheduler == "poly":
        scheduler_power = config["sched_pow"] if config["sched_pow"] is not None else 1
        sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=total_iterations, power=scheduler_power)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iterations, eta_min=0.00001)
    return opt, sched


def load_yaml() -> dict:
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
    assert config['backbone'] is not None, "Please provide a backbone."
    assert config['head'] is not None, "Please provide a head."
    assert config['net'] is not None, "Please provide a net."
    assert config['loss'] is not None, "Please provide a loss."
    assert config['num_classes'] is not None, "Please provide a number of classes."
    backbone = load_net(config['backbone'])
    options["backbone"] = backbone

    head = load_net(config['head'], options={'num_classes': config['num_classes'],'embedding_size':config['embedding_size']})
    options["head"] = head

    options['num_classes'] = config['num_classes']

    if 'collate_fun' not in config:
        collate = None 
    elif config['collate_fun'] == 'channelnorm_collate':
        collate = channelnorm_collate
    # these two collates are the same except that the second one
    # does not load the second image
    elif config['collate_fun'] == 'tuple_channelnorm_collate_dino':
        collate = tuple_channelnorm_collate_dino
    elif config['collate_fun'] == 'tuple_channelnorm_collate_head':
        collate = tuple_channelnorm_collate_head

    net = load_net(config["net"], options)
    loss = load_loss(config["loss"])
    return (net, loss, collate)

# Losser (loss calculator) for self supervised learning with simCLR


def sim_clr_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable):
    x_batch, _, _ = data
    # no transformations
    std_transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float, scale=True)])
    # view for self supervised learning
    transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                   transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                   transforms.ToImage(), transforms.ToDtype(torch.float, scale=True)]
                                   )

    standard_views = std_transform(x_batch).to(device)
    augmented_views = transform(x_batch).to(device)
    block = torch.cat([standard_views, augmented_views], dim=0)
    out_feat = net.forward(block.to(torch.float))
    loss = loss_func(out_feat, device)
    return loss, None

# Losser for supervised learning. Classification of cell types


def perturbations_processing(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable)-> Tuple[torch.Tensor, namedtuple]:
    x_batch, siRNA, metadata = data
    out_feat = net(x_batch.to(device))
    loss = loss_func(out_feat, torch.tensor(siRNA).to(device))
    predicted_labels = out_feat.argmax(dim=1)

    #getting the correct guesses and the total guesses
    correct_labels = (predicted_labels == torch.tensor(siRNA).to(device)).sum().item()
    total_labels = torch.tensor(siRNA).shape[0]

    #putting the results in a named tuple to improve code readability
    AccuracyTuple = namedtuple('acc_tup',['correct_labels','total_labels'])
    accuracy_tuple = AccuracyTuple(correct_labels=correct_labels,total_labels=total_labels)

    return loss, accuracy_tuple


def validation(net, val_loader, device, loss_func, losser, epoch)-> Tuple[List[float],float]:
    """
    Computes the validation loss for a given neural network model.

    Parameters:
    - net: The neural network model.
    - val_loader: DataLoader for the validation dataset.
    - device: The device (CPU/GPU) where computations will be performed.
    - loss_func: The loss function used for evaluation.
    - losser: A function that computes the loss given the model, inputs, and loss function.
    - epoch: The current epoch number (used for logging).

    Returns:
    - validation_loss_values: A list of loss values for each validation batch.
    - accuracy: The total accuracy calculated on the whole testing set
    """

    validation_loss_values = []  # List to store loss values for each batch

    # Create a progress bar for validation
    pbar = tqdm(total=len(val_loader), desc=f"validation-{epoch+1}")

    # Ensure model is in evaluation mode and disable gradient computation
    tot_right_guesses = 0.0
    tot_guesses = 0.0
    net.eval()
    with torch.no_grad():
        for x_batch, siRNA_batch, metadata in val_loader:  # Iterate through validation data
            # Compute loss using the provided 'losser' function
            loss, accuracy_tuple = losser(device, (x_batch, siRNA_batch, metadata), net, loss_func)
            if accuracy_tuple is None:
                raise RuntimeError('You called a validation() function on a losser that doesn\'t output accuracy values!')
            
            # Store the loss value
            validation_loss_values.append(loss.item())

            # Store the right and the total guesses
            tot_right_guesses += accuracy_tuple.correct_labels
            tot_guesses += accuracy_tuple.total_labels

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Validation Loss": loss.item()})

    accuracy = tot_right_guesses / tot_guesses if tot_guesses > 0 else 0.0 # Division by zero paranoia
    return validation_loss_values, accuracy # Return the collected validation loss values



def save_model(name, epoch, net, opt, train_loss, val_loss, batch_size, optimizer, scheduler=None):
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

def sim_clr_processing_norm(device: torch.device, data: tuple, net: torch.nn.Module, loss_func: Callable):
    x_batch, _, _ = data
    out_feat = net.forward(x_batch.to(device))
    loss = loss_func(out_feat, device)
    return loss , None

def vit_small(patch_size=16, in_chans=6, **kwargs):
    """
    Returns a small vision transformer. (Code taken from original repo)
    """
    model = VisionTransformer(
        patch_size=patch_size, in_chans=in_chans, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16, in_chans=3, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def load_dino_weights(model, pretrained_weights, checkpoint_key="student", exclude_projection=True):
    """
    Function to load dino-vit weights. Taken from https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # ========= for 6 channels ===========
        # do not load projection weights since the sizes are mismatching
        if exclude_projection: state_dict.pop("patch_embed.proj.weight",None)
        # ===================================
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please provide a valid file.")
        return None
    return state_dict

import torch
import torch.nn as nn

def initialize_weights(module):
    """
    Recursively initializes the weights of a module and its submodules using Kaiming initialization
    for Linear layers, and uniform initialization for BatchNorm layers.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    
    for submodule in module.children():
        initialize_weights(submodule)


def min_max_scale(tensor, min_val=0, max_val=1):
    """Scales a tensor to the range [min_val, max_val]."""
    X_min, X_max = tensor.min(), tensor.max()
    return min_val + (tensor - X_min) * (max_val - min_val) / (X_max - X_min)
