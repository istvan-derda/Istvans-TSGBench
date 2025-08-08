# István's TSGBench

Derivative Work of TSGBench by Yihao Ang et al. -> https://github.com/YihaoAng/TSGBench

This repository provides code to reproducibly benchmark multiple TSG methods, as suggested by Yihao Ang et al. in their Paper "TSGBench: Time Series Generation Benchmark".

## Setup Development Environment

**Setup on MacOS, Windows or Linux with Dev Containers**

This is the most reproducible option. The repository contains a .devcontainer/devcontainer.json definition. This allows you to open the project with any IDE supporting Dev Containers, and have the same Linux environment as I had when writing the code - on any OS. The instructions below are for a setup with Docker, VS Code and the VS Code Dev Containers extension.

- Install Visual Studio Code
- Follow the installation instructions for the [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- Open the repository in VS Code. It should prompt you if you want to open the project in a devcontainer. 
- After confirming you should get a new VS Code window. This might take a moment to build the container.
- In the bottom left corner it should say "Dev Container: TSG Bench".

**Setup on Linux with conda**

This approach is recommended on systems where you cannot install Docker or don't want to run everything in a container, e.g. on a high performance computing cluster.

- Use any IDE you are comfortable with or just the terminal
- Check if you have conda installed by running `conda` in the terminal.
  - If not: Follow the installation instructions for [miniconda]() or check the documentation of your high performance computing cluster on how to enable conda.
- Create the conda environment by running `conda env create` in the project root and activate it with `conda activate tsg-bench`


## Use TSGBench

**Get the Benchmark Datasets**

`python populate_data.py`

**Run a Benchmark**

Follow the instructions in the subdirectory models/[model you want to benchmark]

**Evaluate a Generated Time Series**

`python evaluate.py <model name>`

The script assumes that a directory `models/<model name>/gen` exists and contains the generated time series data. For details on the expected folder structure and file names see the code.

## Evaluated models:


| Architecture         | Repository                                    | Task                         |
|----------------------|-----------------------------------------------|------------------------------|
| TimeGAN              | https://github.com/jsyoon0823/TimeGAN         | Multivariate TSG             |
| TTS-GAN              | https://github.com/imics-lab/tts-gan          | Multivariate TSG             |
| Time-Transformer AAE | https://github.com/Lysarthas/Time-Transformer | Multivariate TSG             |
| TransFusion          | https://github.com/fahim-sikder/TransFusion   | Multivariate TSG             |


## References

Reference to the original paper by Yihao Ang et al.:

```bibtex
# TSGBench
@article{ang2023tsgbench,
  title        = {TSGBench: Time Series Generation Benchmark},
  author       = {Ang, Yihao and Huang, Qiang and Bao, Yifan and Tung, Anthony KH and Huang, Zhiyong},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {3},
  pages        = {305--318},
  year         = {2023}
}

