from astroclip.utils import download_desi_dataset, load_config
import os


if __name__ == '__main__':
    # 61.1GB dataset, took me roughly 4:40hr to download
    config = load_config()
    cache_dir = config['cache_dir']

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    download_desi_dataset(cache_dir)
