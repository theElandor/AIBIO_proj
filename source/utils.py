from prettytable import PrettyTable
from wilds import get_dataset


def display_configs(configs):
    t = PrettyTable(['Name', 'Value'])
    t.align = "r"
    for key, value in configs.items():
        t.add_row([key, value])
    print(t, flush=True)


def download_dataset():
    dataset = get_dataset(dataset="rxrx1", download=True, root_dir='/work/ai4bio2024/rxrx1/')

