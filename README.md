# Implementation for CMGC

Train a motion prediction model on the Human3.6M dataset.

## Installation
Install the required Python packages:
- Python 3.7+
- torch>=2.0
- timm
- pywt
- ptwt

## Training
To start training on the Human3.6M dataset, run:

python main.py \
  --len_input 10 \
  --len_output 10 \
  --state train \
  --skip_rate 1 \
  --device cuda:0 \
  --dir_path [PATH_TO_DATA] \
  --dct_n 15 \
  --lr 0.0013 \
  --batch_size 256 \
  --is_save=True \
  --epoch 60
