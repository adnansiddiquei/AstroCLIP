# AstroCLIP - Reproduction
This is a reproduction of the AstroCLIP paper by [Parker et al. (2024)](https://arxiv.org/abs/2310.03024) with the original
authors' codebase at [https://github.com/PolymathicAI/AstroCLIP](https://github.com/PolymathicAI/AstroCLIP).

AstroCLIP is a cross-modal, self-supervised foundation model for multi-band galaxy images and optical spectra.
The model creates a shared low-dimensional embedding space which discovers and encodes physically meaningful properties of the
galaxies.
This embedding can be used for both zero-shot and few-shot predictions on a variety of downstream tasks such as redshift
estimation and similarity search.

This repository reproduces the AstroCLIP model using pre-trained image and spectrum encoders from related works, and
trains the model on the dataset provided by [Parker et al. (2024)](https://arxiv.org/abs/2310.03024).
The original AstroCLIP implementation by [Parker et al. (2024)](https://arxiv.org/abs/2310.03024) embeds the images and
spectra into a 512-dimensional space, in this reproduction, we create models to embed the images and spectra into a
a range of low-dimensional spaces: 8, 16, 32, 64, 128, 256, 512.

## Environment setup
This repo uses conda and/or poetry for managing the environment.
If using conda to set up the environment, then run the following commands:
```bash
conda env create -f environment.yml
conda activate astroclip_env
poetry install
```

If using poetry to set up the environment, then simply run the following command:
```bash
poetry install
```

The conda method will create a conda environment and install poetry in it.
Then poetry will install the necessary packages.
The poetry method will just use poetry directly to create a virtual environment and install the necessary packages.
This will install the packages listed in the `pyproject.toml` file.

## Config file
This repository contains a config file [`config.yaml`](config.yaml) which contains a variety of configurations which are loaded by
the scripts utilised in this repository. The config file has been left as is from the original training of this model,
as an example of what it should look like, so feel free to amend it as necessary for you own system.

The first level key is used to define all the locations that you will run the code on. In our case, we have `local` and `hpc`
for our local machine and the Cambridge University High Performance Computing Cluster.

- The `cache_dir` key is used to define where the scripts load data from (dataset, pretrained models);
- The `output_dir` key is used to define where the scripts output results (trained models, logs, etc.);
- The `num_workers` key is used to define the number of workers used in the DataLoader;
- The `pretrained_spectrum_encoder` key defines the filename of the pretrained spectrum encoder, this must be stored in
the `cache_dir` once it has been downloaded;
- The `pretrained_image_encoder` key defines the filename of the pretrained image encoder, this must be stored in
the `cache_dir` as well, once it has been downloaded;

## Training AstroCLIP
### 1. Download the dataset
The AstroCLIP model is trained on a dataset of 197,976 galaxy image-spectra pairs, as provided by
[Parker et al. (2024)](https://arxiv.org/abs/2310.03024).
The multi-band (g,r,z) images were prepared by [Stein et al. (2021)](https://github.com/georgestein/ssl-legacysurvey),
curated from the [Dark Energy Spectroscopic Instrument (DESI)](https://www.desi.lbl.gov/)
Legacy Survey and these images were cross-referenced to find their equivalent optical spectra from the
[DESI Early Data Release](https://www.legacysurvey.org/).

The dataset must be downloaded first before running the code by running:
```bash
python scripts/download_desi_dataset.py --config=local
```
pass in whichever config you wish to use, in this case `local` is used.
This will download the dataset to the directory specified in the `local.cache_dir` key of `config.yaml`.

### 2. Download the pretrained models
AstroCLIP uses two pretrained models for the image and spectrum encoders, both of these pretrained encoders differ from
the ones used in the original paper.
The pretrained spectrum encoder was acquired from the works of [Liang, Melchior et al (2023)](https://github.com/pmelchior/spender)
and can be downloaded by running:
```bash
python scripts/download_pretrained_spectrum_encoder.py --config=local
```
which will download the pretrained spectrum encoder to the directory specified in the `local.cache_dir` key of `config.yaml`.
The pretrained image encoder was taken from the works of [Stein et al. (2021)](https://github.com/georgestein/ssl-legacysurvey)
and can be acquired by following the instructions on the linked repository. Download the `resnet50.ckpt` file from the
Globus endpoint provided by Stein, and place it into the `cache_dir`.

Ensure that both pre-trained models are named exactly as they are specified in their respective keys in the `config.yaml`.

### 3. Generate the spectral standard deviations
Run the following script:
```bash
python scripts/compute_observed_spectra_std_dev.py --config=local
````
to generate `/{output_dir}/observed_spectra_std_dev.pt`, which is a 7781-length tensor of the standard deviations of the
training set spectra at each of the 7781 wavelength bins.
This is used to scale the noise added to each bin of each spectra during training.
See Section (3.2) of the report for more details on this.

### 4. Train the Model
Then to train the model, run:
```bash
python scripts/train_astroclip.py --config=local --jobid=00001 --ckptdir=astroclip_ckpt_00001 --hparams=h01
````

The script uses [WandB](https://wandb.ai/site) for logging so ensure you have an account and are logged in prior to running
the above script.
```bash
wandb login YOUR_API_KEY
```

Otherwise, if you want to disable WandB logging, you can run the script with the additional flag `--no-wandb`:
```bash
python scripts/train_astroclip.py --config=local ... --no-wandb
````

- `--jobid` is used to uniquely identify this run to WandB for logging purposes;
- `--ckptdir` is the name of the directory to store the pytorch Lightning model checkpoints, this directory will be
created inside the `local.output_dir` directory specified in the `config.yaml` file;
- `--hparams` sets the hyperparameters for this run, these are set in the `hyperparameters.yaml` file and
the `h01` hyperparameters are the hyperparameters used in this paper for the 128-dimensional embedding space model.

### Training on a High Performance Computing Cluster (HPC)
See the [`scripts/slurm`](scripts/slurm) directory for the slurm scripts used to run the AstroCLIP model on the Cambridge University HPC.
They are given as used exactly in producing the results in this paper, for the `as3438` HPC user and so will need to be
amended as required, to work on another HPC or user.
The scripts also assume that poetry was used for the environment set up, but this can be amended as necessary.

There are two scripts in the directory, one for the CPU nodes to download the dataset, and one for the GPU nodes to train
the model.
Use a SLURM job to download the dataset as it is a large file and takes some time.
Use `scp` to copy over the pre-trained image and spectrum models to the `cache_dir` on the HPC.
Likewise, use `scp` to copy over `/{output_dir}/observed_spectra_std_dev.pt` to `output_dir` on the HPC.
Then you can submit a job to run the training script.

## Downstream tasks
See the [`results/downstream_tasks/`](results/downstream_tasks) folder for the scripts and notebooks used to generate the results and plots in the paper
for the downstream tasks.

## Training Statistics
See the [`results/training`](results/training) folder for the scripts, notebooks and output logs from Weights and Biases
used to generate the training related statistics and plots in the paper.

## AstroCLIP Model Weights
The Lightning model checkpoints for all 7 trained AstroCLIP models used in the paper are available on HuggingFace
[here](https://huggingface.co/adnanshirik/astroclip). Given that this piece of work is assessed, the model page on
HuggingFace is gated and access will need to be requested until the assessment is complete.

If you use one of these trained models, you can copy the model weights directly into the `cache_dir` and follow the
instructions in [`results/downstream_tasks/`](results/downstream_tasks) to reproduce the results of the
downstream tasks for the trained model.

## Report
The report and executive summary for this reproduction is available in the [`report`](report) folder.
- [Report](report/main.pdf)
- [Executive Summary](report/exec-summary.pdf)

## Acknowledgements
I would primarily like to thank [Miles Cranmer](https://github.com/MilesCranmer) for his guidance on this project.

The dataset used in this paper was prepared entirely by [Parker et al. (2024)](https://arxiv.org/abs/2310.03024)
and the [`astroclip/legacy_survey.py`](astroclip/legacy_survey.py) file was taken from their repository to load the dataset.
The pretrained spectrum encoder was acquired from the works of [Liang, Melchior et al (2023)](https://github.com/pmelchior/spender)
and the pretrained image encoder was acquired from the works of [Stein et al. (2021)](https://github.com/georgestein/ssl-legacysurvey).
The `ToRGB` function used for plotting the images was taken as is (with minor modifications) from [legacysurvey](https://github.com/legacysurvey/imagine).

## Use of Auto-Generation tools
GitHub co-pilot use was minimal, and was limited to using the auto-complete for generating documentation for some functions
and classes.
