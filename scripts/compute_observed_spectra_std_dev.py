from astroclip.utils import (
    download_desi_dataset,
    SpectralStdDevCalculator,
    format_axes,
    save_fig,
    load_config,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from astroclip.transforms import ExtractKey, Permute, Standardise
import matplotlib.pyplot as plt
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

    # load the dataset from cache
    dataset = download_desi_dataset(cache_dir)

    transforms = nn.Sequential(
        ExtractKey('spectrum'),
        Permute([0, 2, 1]),
        Standardise(return_mean_and_std=False),
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
    fig, ax = plt.subplots(figsize=(10, 3))

    plt.plot(
        observed_spectra_std_dev.squeeze(),
        label='AstroCLIP Reproduction',
        linestyle='-',
    )
    plt.xlabel(r'Spectrum Index $i$')
    plt.ylabel(r'Standard Deviation $\sigma_{sp}(i)$')
    plt.xlim(-50, 7781 + 50)
    format_axes(ax)

    save_fig(output_dir, 'observed_spectra_std_dev.png')


if __name__ == '__main__':
    main()
