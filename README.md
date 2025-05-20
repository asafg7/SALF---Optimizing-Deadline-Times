# Adaptive Deadline and Batch Layered Synchronized Federated Learning

## Introduction
We introduce ADEL-FL, a federated learning algorithm that optimizes per-round deadlines and batch sizes to accelerate convergence under time constraints. This codebase contains a PyTorch implementation of ADEL-FL.

## Usage
This code has been tested on Python 3.11

### Prerequisite
1. PyTorch=2.5.1
2. scipy=1.14.1
3. tqdm=4.66.5
4. matplotlib=3.9.2
5. torchinfo=1.8.0
6. TensorboardX=2.6.2.2

### Training
```
python3 main.py --exp_name=adel-fl_cnn_mnist --data mnist --model cnn2 --global_epochs 50 --t_max 200 --num_users 20 --lr 0.5
```

### Testing
```
python3 main.py --exp_name=adel-fl_cnn_mnist --eval
```
