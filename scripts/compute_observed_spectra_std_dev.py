from astroclip.utils import (
    download_desi_dataset,
    SpectralStdDevCalculator,
    format_axes,
    save_fig,
    load_config,
)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from astroclip.transforms import ExtractKey, Permute, Standardize
import matplotlib.pyplot as plt
import numpy as np
import argparse


@torch.no_grad()
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
    output_dir = config['output_dir']

    if not os.path.exists(cache_dir):
        raise FileNotFoundError(
            f'Cache directory {cache_dir} does not exist. Please download the dataset first.'
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load the dataset from cache
    dataset = download_desi_dataset(cache_dir)

    transforms = nn.Sequential(
        ExtractKey('spectrum'),
        Permute([0, 2, 1]),
        Standardize(return_mean_and_std=False),
    )

    dataloader = DataLoader(dataset['train'], batch_size=1024, num_workers=4)

    # Compute and save the observed spectra standard deviation
    ssc = SpectralStdDevCalculator(dataloader, transforms)

    observed_spectra_std_dev = ssc.compute()

    torch.save(observed_spectra_std_dev, f'{output_dir}/observed_spectra_std_dev.pt')

    print(
        f'Observed spectra standard deviation computed and saved to {output_dir}/observed_spectra_std_dev.pt'
    )

    # Plot and save the above tensor
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(
        observed_spectra_std_dev.squeeze(),
        label='AstroCLIP Reproduction',
        linestyle='-',
    )
    plt.xlabel('Spectrum Index')
    plt.ylabel('Standard Deviation')
    format_axes(ax)

    # If it exists in the cache dir, load the original paper's standard deviation and plot it side by side with
    # the reproduced one
    if os.path.exists(f'{cache_dir}/spectra_std.npz'):
        std_spectra = np.load(f'{cache_dir}/spectra_std.npz')['spectra_npz'].astype(
            'float32'
        )
        plt.plot(
            std_spectra, label='AstroCLIP Original Paper (Lanusse et al, 2023)', lw=0.5
        )
        plt.legend()

    save_fig(output_dir, 'observed_spectra_std_dev.png')


if __name__ == '__main__':
    main()
