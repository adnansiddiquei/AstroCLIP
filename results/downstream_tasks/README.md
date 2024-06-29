# Downstream tasks

This directory contains the Jupyter notebooks used to generate the results and plots for the downstream tasks:
- Zero-shot redshift predictions
- Similarity search

To run the notebooks, you will need to do the following:
1. Follow all the instructions in the main README.md file to set up the environment, download the data and train the
AstroCLIP model.
2. Move the trained AstroCLIP model to the cache directory.
I.e., move `out/astroclip_ckpt_{jobID}/astroclip-epoch={epoch}-min.ckpt` (where the `train_astroclip.py` script will
naturally output the lowest validation loss model) to `{cache_dir}/astroclip_model.ckpt` (feel free to call it whatever
you want, in this case I have called it `astroclip_model.ckpt`).
3. Run `python generate_embeddings.py --config=local --model-checkpoint=astroclip_model.ckpt` to generate the trained
embeddings. Ensure to pass in the correct model checkpoint name (exactly as it is saved in the cache directory).
The script will output the following files to the cache directory:
    - `spectrum_embeddings.pt`
    - `image_embeddings.pt`
    - `valid_indices.pt`
4. You can now run through the notebooks to generate the results of the downstream tasks and the plots used in the paper.

**What is the `valid_indices.pt`?**: There are 39,599 image-spectra pairs in the validation set.
The `generate_embeddings.py` script will therefore generate a tensor of shape (39599, 128) for the spectrum and image embeddings.
`valid_indices.pt` will be of shape (39219,) which is exactly 380 elements shorter than the embedding tensors.
This is because there are 380 image-spectra pairs that have been removed from the validation set due to invalid spectra
or redshift being outside the range [0, 0.8] (see Section (3.4) of the report)
These invalid pairs are removed on the fly at the start of each minibatch during training, they are not removed directly
from the dataset (see Section (5) of the report).
Therefore, `valid_indices.pt` contains the indices of the valid rows in `spectrum_embeddings.pt`, `image_embeddings.pt`.
The reason that we do not just generate the embeddings with these invalid rows removed is because we want to maintain the
correspondence between the rows in the embeddings and the original validation set.
For example, we want row 628 in the embeddings to correspond to the 628th image-spectrum pair in the validation set,
otherwise it makes it difficult to match the embeddings to their respective image-spectra pairs and redshift data.
