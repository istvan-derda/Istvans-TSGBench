# TimeTransformer

## Init

`source env/default/init.bash`

This will:
- Pull a pinned version of the TimeTransformer from GitHub
- Set up a conda environment with necessary dependencies
- Enter that conda environment

For running on the high performance cluster of leipzig university, instead run:

`source env/sc_uni_leipzig/init.bash`

## Run

Choose which dataset to use e.g.:

`export TSGB_USE_DATASET=D5_energy`

Then run the TimeTransformer

`python run_time_transformer.py`