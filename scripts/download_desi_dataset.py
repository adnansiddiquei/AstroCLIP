from astroclip.utils import download_desi_dataset
import yaml
import os


if __name__ == '__main__':
    # 61.1GB dataset, took me roughly 4:40hr to download
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    cache_dir = config['general']['cache_dir']

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    download_desi_dataset(cache_dir)
