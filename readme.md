# Emoeji: Emoji Prediction Project

## Overview
This project focuses on emoji prediction using natural language processing techniques.

## Installation

### Prerequisites
- Git
- Conda
- CUDA-compatible GPU (optional, for faster training)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/zzzlou/CS4248-project.git
cd CS4248-project
```

2. Create and activate conda environment:
```bash
conda create -n emoeji
conda activate emoeji
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Configuration (Optional)
If you need to install PyTorch with a specific CUDA version:
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Additional packages if needed:
```bash
pip install pandas transformers emoji accelerate matplotlib left scikit-learn wonderwords
```

Set up GPU for training:
```bash
export CUDA_VISIBLE_DEVICES=0
```

## Usage

### Data Preparation
```bash
cd scripts/dataset
python data_from_ELCo.py
python get_descriptions.py
python data_generating.py
```

### Training
```bash
cd scripts
python train.py
```


