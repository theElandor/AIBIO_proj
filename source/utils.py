import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prettytable import PrettyTable
from wilds import get_dataset
import torch
import torch.nn.functional as F
import os
from source.net import SimCLR
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from scipy.stats import mode



def display_configs(configs):
    t = PrettyTable(['Name', 'Value'])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def load_device(config):
    if config['device'] == 'gpu':
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = 'cuda:0'
        print("Training network on {}".format(torch.cuda.get_device_name(device=device)))
    else:
        device = torch.device('cpu')
    return device


def download_dataset():
    dataset = get_dataset(dataset="rxrx1", download=True, root_dir='/work/ai4bio2024/rxrx1/')


def contrastive_loss(features, device, temperature=0.5):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.mm(features, features.T) / temperature
    batch_size = features.shape[0]
    labels = torch.arange(batch_size).to(device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def config_loader(config):
    net, loss, opt = ..., ..., ...
    if str(config['net']).__contains__("simclr"):
        net = SimCLR()

    if str(config['loss']).__contains__("contrastive"):
        loss = contrastive_loss

    if str(config['opt']).__contains__("adam"):
        opt = torch.optim.Adam(net.parameters(), lr=0.005)

    return (net, loss, opt)


def test_kmean_accuracy(net, test_loader, device):
    net.eval()
    test_features = []
    test_labels = []
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(test_loader):  # Suppongo tu abbia le etichette nel test set
            features = net(x_batch.to(torch.float).to(device))  # Estrazione delle feature
            test_features.append(features)
            test_labels.append(y_batch)

    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)
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


def validation_loss(net, val_loader, device, transform, loss_func):
    validation_loss_values = []
    pbar = tqdm(total=len(val_loader), desc=f"validation")
    with net.eval() and torch.no_grad():
        for x_batch, _, _ in val_loader:
            augmented_views = torch.cat([transform(img.unsqueeze(0)) for img in x_batch], dim=0).to(device)

            out_feat = net.forward(augmented_views.to(torch.float))
            loss = loss_func(out_feat, device)

            validation_loss_values.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'Validation Loss': loss.item()})

    return validation_loss_values


def save_model(epoch, net, opt, train_loss, val_loss, batch_size, checkpoint_dir, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'training_loss_values': train_loss,
        'validation_loss_values': val_loss,
        'batch_size': batch_size,
        'optimizer': optimizer,
    }, os.path.join(checkpoint_dir, "checkpoint{}".format(epoch + 1)))
