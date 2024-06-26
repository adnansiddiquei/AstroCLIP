from astroclip.utils import load_config
import os
import argparse
import spender
import torch


def main():
    # load the config argument to decide which config config_header to load
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

    # show list of pretrained models
    spender.hub.list()

    # print out details for SDSS model from paper II
    print(spender.hub.help('sdss_II'))

    # if your machine does not have GPUs, specify the device
    sdss, model = spender.hub.load('desi_edr_galaxy', map_location='cpu')

    # save the model into the cache_dir
    torch.save(model.encoder.state_dict(), f'{cache_dir}/spender.desi-edr.encoder2.pth')


if __name__ == '__main__':
    main()
