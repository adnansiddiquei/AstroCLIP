from astroclip.utils import download_desi_dataset, load_config
import os
import argparse


if __name__ == '__main__':
    # load the config argument to decide which config header to load
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='Which config to use.',
        default='local',
    )
    args = parser.parse_args()

    # load the config file
    config = load_config(args.config)

    cache_dir = config['cache_dir']

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    download_desi_dataset(cache_dir)
