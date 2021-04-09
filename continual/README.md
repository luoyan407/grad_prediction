# Learning to Predict Gradients for Semi-Supervised Continual Learning
Code and script for experiments on MNIST-R, MNIST-P, and iCIFAR-100. 

## Environment and Requirements

* The code is built upon [DCL](https://github.com/luoyan407/congruency) and tested with a Nvidia 2080 ti GPU on Ubuntu 1804.

* Install [EfficientNet PyTorch (0.6.3+)](https://github.com/lukemelas/EfficientNet-PyTorch)
```bash
pip install efficientnet_pytorch
```
* Download [continual learning datasets](https://drive.google.com/drive/folders/1jFeKzrjQj6vjzMICLMCS-A-Lf19a2X-v?usp=sharing), and then move the pt files to **./data/**. Note that GEM provides the script building datasets if running the experiment the first time. To reproduce the performance reported in this work, please use the shared pre-generated datasets instead.

* Download Tiny ImageNet by running
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

## Usage

All the experiments are archived in script file **archive_experiments.sh**. In this script, modify variable **extra-data** to point to your local Tiny ImageNet dataset. Then, run  
```bash
./archive_experiments.sh
```

## Performance and Pretrained Models
For convenience, script **archive_experiments.sh** contains both the command to run each experiment and the corresponding performance. The pretrained models can be found in this shared google drive [folder](https://drive.google.com/drive/folders/1XScv5B_4SYph63BWN_N9AG6-dm8aI7Vz?usp=sharing).