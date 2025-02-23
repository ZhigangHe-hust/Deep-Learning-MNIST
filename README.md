<p align="center">
    <h1 align="center">Deep Learning Practice: MNIST Handwritten Digit Recognition</h1>
</p>


<br>
<div align="center">

English | [简体中文](README_CN.md)

</div>

<p align="center">
    🌟This project uses<strong> Fully connected neural network</strong>Complete the task of classifying handwritten digits on the <strong>MNIST dataset</strong>, based on <strong>Pytorch</strong>. This project can be used as an entry practice for deep learning project development based on Pytorch.
</p>

# Installation

## Dependencies

```
Python 3.10
Pytorch 2.1.2
CUDA 12.1
torchvission 0.16.2
```

## Environment Installation

### 1.Create the environment and activate it

```
conda create --name mnist python=3.10
conda activate mnist
```

You can view the version information of Python, Pytorch, and CUDA by running env_info.py

```
python env.py
```

The results are as follows (reference):
![image](https://github.com/ZhigangHe-hust/Deep-Learning-MNIST/blob/main/figs/fig2.png)

### 2.Install Pytorch and CUDA

Pytorch2.1.2 and CUDA11.8 are installed based on RTX3060. You can install the appropriate version of Pytorch and CUDA according to your graphics card model.

```
conda install pytorch=2.1.2 torchvision cudatoolkit=11.8 -c pytorch -c nvidia
```

### 3.Install other dependencies

```
conda install tqdm
conda install matplotlib
pip install numpy==1.23.5
```

# Dataset preparation

## Introduction to the MNIST Dataset

### Overview

**MNIST (Modified National Institute of Standards and Technology)** is a classic handwritten digit image dataset, widely used in introductory tutorials and benchmarks in the field of machine learning and deep learning.

### Dataset Contents

**Number of images**: 70,000 grayscale images <br>
**Training dataset**: 60,000 images <br>
**Test dataset**: 10,000 images <br>
**Image format**: 28x28 grayscale images <br>
**Labels**: Each image has a number label from 0 to 9 <br>

### Data Example

Here is some data from the MNIST dataset:
![image](https://github.com/ZhigangHe-hust/Deep-Learning-MNIST/blob/main/figs/fig1.png)

## Download the dataset

MNIST is already integrated into Pytorch, so you can download it directly through the script

```
python data.py
```

# Model Training

Before training, you need to create a folder in the current directory to store the checkpoint file

```
├─ data.py
├─ dataloader.py
├─ ···
├─ output
│ ├─ checkpoint
```
Input in the terminal
```
python train.py
```

If model training terminates unexpectedly, or you want to add more training rounds, you can run the following command:

```
python retrain.py --checkpoint_path output\checkpoint\checkpoint_epoch_N.pth --epochs M
# M is the total number of training rounds for the final model (default is 10)
# checkpoint_epoch_N is your last checkpoint file
```

# Model Testing

```
python test.py
```

# Contact Us

If you have any other questions❓, please contact us in time 👬
