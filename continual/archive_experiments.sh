#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MY_PYTHON="python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"
# build datasets if running the experiment the first time, 
# to reproduce the performance reported in this work, please use the shared pre-generated datasets instead.
: "
# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_rotations.py \
	--o mnist_rotations.pt\
	--seed 0 \
	--min_rot 0 \
	--max_rot 180 \
	--n_tasks 20

$MY_PYTHON mnist_permutations.py \
	--o mnist_permutations.pt \
	--seed 0 \
	--n_tasks 20

$MY_PYTHON cifar100.py \
	--o cifar100.pt \
	--seed 0 \
	--n_tasks 20

cd ..
"
#-----------------MNIST-R-----------------
# perf: 0.8303 -0.0061 0.6482
$MY_PYTHON main.py $MNIST_ROTA \
	--model gem --lr 0.1 \
	--n_memories 256 --memory_strength 0.7 
# perf: 0.8654 0.0227 0.6537
$MY_PYTHON main_gp.py $MNIST_ROTA \
	--model gem_gp --lr 0.1 \
	--n_memories 256 --memory_strength 0.7 \
	--extra-data-name 'tiny4mnist' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--extra-input-size 28 \
	--gl_dim 10 \
	--gl_prob 0.15 \
	--gl_scale 0.001 \
	--gl_arch 64 16 \
	--gl_loss_scale 0.3 \
	--gl_start_predict 0

# perf: 0.8488 0.0088 0.6526
$MY_PYTHON main.py $MNIST_ROTA \
	--model dcl --lr 0.1 --n_memories 256 \
	--memory_strength 0.7 --reset_interval 10 
# perf: 0.8626 0.0106 0.6620
$MY_PYTHON main_gp.py $MNIST_ROTA \
	--model dcl_gp --lr 0.1 --n_memories 256 \
	--memory_strength 0.7 --reset_interval 10 \
	--extra-data-name 'tiny4mnist' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--extra-input-size 28 \
	--gl_dim 10 \
	--gl_prob 0.15 \
	--gl_scale 0.001 \
	--gl_arch 64 16 \
	--gl_loss_scale 0.3 \
	--gl_start_predict 0
#-----------------MNIST-P-----------------
# perf: 0.8235 0.0251 -0.0101
$MY_PYTHON main.py $MNIST_PERM \
	--model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
# perf: 0.8291 0.0316 -0.0072
$MY_PYTHON main_gp.py $MNIST_PERM \
	--model gem_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--extra-data-name 'tiny4mnist' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--extra-input-size 28 \
	--gl_dim 10 \
	--gl_prob 0.15 \
	--gl_scale 0.001 \
	--gl_arch 64 16 \
	--gl_loss_scale 0.5 \
	--gl_start_predict 0

# perf: 0.8283 0.0279 -0.0100
$MY_PYTHON main.py $MNIST_PERM \
	--model dcl --lr 0.1 --n_memories 256 \
	--memory_strength 0.5 --reset_interval 4
# perf: 0.8297 0.0402 -0.0038
$MY_PYTHON main_gp.py $MNIST_PERM \
	--model dcl_gp --lr 0.1 --n_memories 256 \
	--memory_strength 0.5 --reset_interval 4 \
	--extra-data-name 'tiny4mnist' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--extra-input-size 28 \
	--gl_dim 10 \
	--gl_prob 0.1 \
	--gl_scale 0.001 \
	--gl_arch 64 16 \
	--gl_loss_scale 0.49 \
	--gl_start_predict 0
#-----------------iCIFAR-100-------------------------
# perf: 0.6692, 0.0132, -0.0048
$MY_PYTHON main.py $CIFAR_100i \
	--model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
# perf: 0.6874 0.0619 0.0055
$MY_PYTHON main_gp.py $CIFAR_100i \
	--model gem_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--extra-data-name 'tinyimagenet' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--gl_dim 5 \
	--gl_prob 0.3 \
	--gl_scale 0.005 \
	--gl_arch 128 32 \
	--gl_loss_scale 2.0 \
	--gl_start_predict 50

# perf: 0.6755 0.0048 -0.0117
$MY_PYTHON main.py $CIFAR_100i \
	--model dcl --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--reset_interval 4 --backend_net 'resnet'
# perf: 0.6853 0.0574 -0.0038
$MY_PYTHON main_gp.py $CIFAR_100i \
	--model dcl_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--reset_interval 5 --backend_net 'resnet' \
	--extra-data-name 'tinyimagenet' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--gl_dim 5 \
	--gl_prob 0.3 \
	--gl_scale 0.005 \
	--gl_arch 128 32 \
	--gl_loss_scale 2.5 \
	--gl_start_predict 50

# perf: 0.8144 0.0128 0.0105
$MY_PYTHON main.py $CIFAR_100i \
	--model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --backend_net 'efficientnet-b1'
# perf: 0.8551 0.0219 0.0148
$MY_PYTHON main_gp.py $CIFAR_100i \
	--model gem_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--backend_net 'efficientnet-b1' \
	--extra-data-name 'tinyimagenet' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--gl_dim 5 \
	--gl_prob 0.2 \
	--gl_scale 0.005 \
	--gl_arch 128 32 \
	--gl_loss_scale 2.0 \
	--gl_start_predict 50

# perf: 0.8347 0.0266 -0.0185
$MY_PYTHON main.py $CIFAR_100i \
	--model dcl --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--reset_interval 4  --backend_net 'efficientnet-b1'
# perf: 0.8684 0.0257 -0.0187
$MY_PYTHON main_gp.py $CIFAR_100i \
	--model dcl_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--reset_interval 4  --backend_net 'efficientnet-b1' \
	--extra-data-name 'tinyimagenet' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--gl_dim 5 \
	--gl_prob 0.15 \
	--gl_scale 0.005 \
	--gl_arch 128 32 \
	--gl_loss_scale 2.0 \
	--gl_start_predict 50
# perf: 0.8570 0.0378 0.0017
$MY_PYTHON main_gp.py $CIFAR_100i \
	--model dcl_gp --lr 0.1 --n_memories 256 --memory_strength 0.5 \
	--reset_interval 4  --backend_net 'efficientnet-b1' \
	--extra-data-name 'tinyimagenet' \
	--extra-data '/path/to/tinyimagenet/' \
	--extra-batch 4 \
	--gl_dim 5 \
	--gl_prob 0.35 \
	--gl_scale 0.005 \
	--gl_arch 128 32 \
	--gl_loss_scale 2.0 \
	--gl_start_predict 50