import os, sys
import torchvision.transforms.v2 as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prettytable import PrettyTable
from wilds import get_dataset
import torch
import torch.nn.functional as F
import os
from source.net import SimCLR
from source.net import FcHead
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from scipy.stats import mode
from typing import Callable


def display_configs(configs):
    t = PrettyTable(["Name", "Value"])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def load_device(config):
    if config["device"] == "gpu":
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = "cuda:0"
        print(
            "Training network on {}".format(torch.cuda.get_device_name(device=device))
        )
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

    else:
        device = torch.device("cpu")
    return device


def download_dataset():
    dataset = get_dataset(dataset="rxrx1", download=True, root_dir="")


def contrastive_loss(features, device, temperature=0.5):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.mm(features, features.T) / temperature
    batch_size = features.shape[0]
    labels = torch.arange(batch_size).to(device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def info_nce_loss(features, device, temperature=0.5):
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
        return SimCLR()
    if netname == "fc_head":
        return FcHead(num_classes=options["num_classes"])
    else:
        raise ValueError("Invalid netname")

def load_loss(lossname: str) -> Callable:
    if lossname == "contrastive":
        return contrastive_loss
    elif lossname == "NCE":
        return info_nce_loss
    else:
        raise ValueError("Invalid lossname")

def load_opt(optimizer: str, net: torch.nn.Module) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=0.005)
    else:
        raise ValueError("Invalid optimizer")

def config_loader(config):
    net = load_net(config["net"])
    loss = load_loss(config["loss"])
    opt = load_opt(config["opt"], net)
    return (net, loss, opt)


def sim_clr_processing(device: torch.device, x_batch: torch.Tensor, net: torch.nn.Module, loss_func: Callable):
    # no transformations
    std_transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    # view for self supervised learning
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    ])
    standard_views = torch.cat(
        [std_transform(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
    augmented_views = torch.cat(
        [transform(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)
    block = torch.cat([standard_views, augmented_views], dim=0)
    out_feat = net.forward(block.to(torch.float))
    loss = loss_func(out_feat, device)
    return loss


def test_kmean_accuracy(net, test_loader, device):
    net.eval()
    test_features = []
    test_labels = []
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(
            test_loader
        ):  # Suppongo tu abbia le etichette nel test set
            features = net(
                x_batch.to(torch.float).to(device)
            )  # Estrazione delle feature
            test_features.append(features)
            test_labels.append(y_batch.to(device))

    test_features = torch.cat(test_features).cpu()
    test_labels = torch.cat(test_labels).cpu()
    kmeans = KMeans(n_clusters=4, random_state=42)
    predicted_clusters = kmeans.fit_predict(test_features)
    cluster_to_class = {}

    for cluster_id in range(4):
        indices = np.where(predicted_clusters == cluster_id)[0]
        true_labels = test_labels[indices]
        most_common_class = mode(true_labels).mode
        cluster_to_class[cluster_id] = most_common_class

    mapped_predictions = np.array([cluster_to_class[c] for c in predicted_clusters])
    accuracy = np.mean(mapped_predictions == test_labels.numpy())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def validation_loss(net, val_loader, device, loss_func):
    validation_loss_values = []
    pbar = tqdm(total=len(val_loader), desc=f"validation")
    with net.eval() and torch.no_grad():
        for x_batch, _, _ in val_loader:
            loss = sim_clr_processing(device, x_batch, net, loss_func)
            validation_loss_values.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({"Validation Loss": loss.item()})

    return validation_loss_values


def save_model(epoch, net, opt, train_loss, val_loss, batch_size, checkpoint_dir, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "training_loss_values": train_loss,
            "validation_loss_values": val_loss,
            "batch_size": batch_size,
            "optimizer": optimizer,
        },
        os.path.join(checkpoint_dir, "checkpoint{}".format(epoch + 1)),
    )
