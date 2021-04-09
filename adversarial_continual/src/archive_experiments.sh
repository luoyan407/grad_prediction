#!/bin/bash
python main.py --config ./configs/config_cifar100.yml
python main_gp.py --config ./configs/config_cifar100_gp.yml
python main.py --config ./configs/config_miniimagenet.yml
python main_gp.py --config ./configs/config_miniimagenet_gp.yml