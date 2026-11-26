# Official implementations of LaMP
From Language to Prompts: Multi-Path Prompt Learning for Zero-Shot Domain Generalization

# Installation

```bash
conda create -n lamp python=3.10 -y
conda activate lamp
python -m pip install --upgrade pip
python -m pip install -r requirment.txt
python -m pip check
```

# Datasets

Set a common dataset root, for example:

```bash
export DATASET_ROOT=/absolute/path/to/Data4DG
mkdir -p "$DATASET_ROOT"
```

# Training and Testing

Run from the project root, for example:

```bash
# ResNet-50 PACS
bash scripts/pacs_resnet50.sh
```

# Acknowledgement
This project is built on the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) framework. Special thanks are extended to the authors of that project.
