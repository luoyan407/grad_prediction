# Learning to Predict Gradients for Semi-Supervised Continual Learning
Code and script for experiments of adversarial continual learning on CIFAR-100 and MiniImageNet. 

## Environment and Requirements

* The code is built upon [ACL](https://github.com/facebookresearch/Adversarial-Continual-Learning) and tested with a Nvidia 2080 ti GPU on Ubuntu 1804.

* Download [continual CIFAR-100 and MiniImageNet](https://drive.google.com/drive/folders/1We3sLW-USiNpq2Ci8p48kTm-u-vxXDOo?usp=sharing), and then move folders cifar-100-python and miniimagenet to **adversarial_continual/data/**. Note that ACL does not provide the script building MiniImageNet. We provide [data/generate_pickle.py](data/generate_pickle.py) to generate MiniImageNet for adversarial continual learning.

* Download Tiny ImageNet by running
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

## Usage

All the experiments are archived in script file **archive_experiments.sh**. Please modify variable **extra_data** in [src/configs/config_cifar100_gp.yml](src/configs/config_cifar100_gp.yml) or [src/configs/config_miniimagenet_gp.yml](src/configs/config_miniimagenet_gp.yml) to point to your local extra datasets. Then, run  
```bash
./archive_experiments.sh
```

## Performance and Pretrained Models
The pretrained models can be found in this shared google drive [folder](https://drive.google.com/drive/folders/1MBOEW7aMErT6Ja0X2ybjlQE3M8_4RAeI?usp=sharing).