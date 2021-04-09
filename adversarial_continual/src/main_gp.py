import os,argparse,time
import numpy as np
from omegaconf import OmegaConf

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils

import torch.utils.data as torchdata
import torchvision
import torchvision.transforms as torchtransforms
import torchvision.datasets as torchdatasets
from torchvision.datasets import ImageFolder

tstart=time.time()


# Arguments
parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_mnist5.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()


########################################################################################################################

# Args -- Experiment
if args.experiment=='pmnist':
    from dataloaders import pmnist as datagenerator
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as datagenerator
elif args.experiment=='cifar100':
    from dataloaders import cifar100 as datagenerator
elif args.experiment=='miniimagenet':
    from dataloaders import miniimagenet as datagenerator
elif args.experiment=='multidatasets':
    from dataloaders import mulitidatasets as datagenerator
else:
    raise NotImplementedError

from acl_gp import ACL as approach

# Args -- Network
if args.experiment == 'mnist5' or args.experiment == 'pmnist':
    from networks import mlp_acl as network
elif args.experiment == 'cifar100' or args.experiment == 'miniimagenet' or args.experiment == 'multidatasets':
    from networks import alexnet_acl as network
else:
    raise NotImplementedError

import fg_datasets

########################################################################################################################

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    return d_tr, d_te, n_inputs, n_outputs.item() + 1, len(d_tr)

class GradLearner(nn.Module):
    def __init__(self, dim, gal_arch=[64],
            gl_scale=0.01, gl_loss_scale=1.0):
        super(GradLearner, self).__init__()
        self.dim = dim
        self.gal_arch = gal_arch
        self.gl_scale = gl_scale
        self.gl_loss_scale = gl_loss_scale

        self.net = None
        if gal_arch is not None:
            net = []
            tmp_outs = gal_arch

            needBias = True
            stride_val = 2
            kernal_size = 3

            in_channel = self.dim
            for i in range(0,len(gal_arch)):
                net.append(nn.Linear(in_channel, gal_arch[i]))
                in_channel = gal_arch[i]
            net.append(nn.Linear(in_channel, self.dim))

            self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x

def run(args, run_id):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        # Faster run but not deterministic:
        # torch.backends.cudnn.benchmark = True

        # To get deterministic results that match with paper at cost of lower speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loader
    print('Instantiate data generators and model...')
    dataloader = datagenerator.DatasetGen(args)
    args.taskcla, args.inputsize = dataloader.taskcla, dataloader.inputsize
    if args.experiment == 'multidatasets': args.lrs = dataloader.lrs

    # Model
    net = network.Net(args)
    net = net.to(args.device)

    net.print_model_size()

    grad_learner = GradLearner(args.gl_dim, args.gl_arch, 
                gl_scale=args.gl_scale, gl_loss_scale=args.gl_loss_scale)
    grad_learner = grad_learner.to(args.device)
    output_str = '====>Total learner params: {:.2f}M + {}'.format(
            sum(p.numel() for p in net.parameters())/1000000.0,
            sum(p.numel() for p in grad_learner.parameters()))
    print(output_str)

    # Approach
    appr=approach(net,args,network=network, 
                extradata=None, gl_start_predict=args.gl_start_predict, gl_prob=args.gl_prob,
                extra_data_name=args.extra_data_name, grad_learner=grad_learner)

    # Loop tasks
    acc=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    lss=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)

    for t,ncla in args.taskcla:

        x_in_size = args.extra_input_size
        extradata = None
        if 'tiny' in args.extra_data_name:
            extra_data_dir = os.path.join(args.extra_data, 'train')
            normalize = torchtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            extra_data_loader = torchdata.DataLoader(
                torchdatasets.ImageFolder(extra_data_dir, torchtransforms.Compose([
                    torchtransforms.RandomCrop((60,60)),
                    torchtransforms.Resize(x_in_size),
                    torchtransforms.RandomHorizontalFlip(),
                    torchtransforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.extra_batch, shuffle=True,
                num_workers=2)
            extradata = iter(cycle(extra_data_loader))
        elif 'coco' in args.extra_data_name:
            normalize = torchtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            extra_tfms = torchtransforms.Compose(
                    [torchtransforms.Resize(256),
                     torchtransforms.RandomCrop(224),
                     torchtransforms.Resize(x_in_size),
                     torchtransforms.RandomHorizontalFlip(),
                     torchtransforms.ToTensor(),
                     normalize,
                     ])
            extra_dataset = ImageFolder(args.extra_data, extra_tfms)
            extra_data_loader = torch.utils.data.DataLoader(
                extra_dataset, batch_size=args.extra_batch, shuffle=True,
                num_workers=2, pin_memory=True, drop_last=False)
            extradata = iter(cycle(extra_data_loader))
        elif 'cub' in args.extra_data_name:
            extraset = fg_datasets.CUB(input_size=x_in_size, root=args.extra_data, is_train=True)
            extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                                  shuffle=True, num_workers=2, drop_last=False)
            extradata = iter(cycle(extra_data_loader))
        elif 'car' in args.extra_data_name:
            dataset_name = 'car'
            extraset = fg_datasets.STANFORD_CAR(input_size=x_in_size, root=args.extra_data, is_train=True)
            extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                                  shuffle=True, num_workers=2, drop_last=False)
            extradata = iter(cycle(extra_data_loader))
        elif 'aircraft' in args.extra_data_name:
            dataset_name = 'Aircraft'
            extraset = fg_datasets.FGVC_aircraft(input_size=x_in_size, root=args.extra_data, is_train=True)
            extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                                  shuffle=True, num_workers=2, drop_last=False)
            extradata = iter(cycle(extra_data_loader))
        appr.extradata = extradata

        print('*'*250)
        dataset = dataloader.get(t)
        print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
        print('*'*250)

        # Train
        trval_stats = appr.train(t,dataset[t])
        print('-'*250)
        print()

        for u in range(t+1):
            # Load previous model and replace the shared module with the current one
            test_model = appr.load_model(u)
            test_res = appr.test(dataset[u]['test'], u, model=test_model)

            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, dataset[u]['name'],
                                                                                          test_res['loss_t'],
                                                                                          test_res['acc_t']))


            acc[t, u] = test_res['acc_t']
            lss[t, u] = test_res['loss_t']


        # Save
        print()
        print('Saved accuracies at '+os.path.join(args.checkpoint,args.output))
        np.savetxt(os.path.join(args.checkpoint,args.output),acc,'%.6f')

    # Extract embeddings to plot in tensorboard for miniimagenet
    if args.tsne == 'yes' and args.experiment == 'miniimagenet':
        appr.get_tsne_embeddings_first_ten_tasks(dataset, model=appr.load_model(t))
        appr.get_tsne_embeddings_last_three_tasks(dataset, model=appr.load_model(t))

    avg_acc, gem_bwt = utils.print_log_acc_bwt(args.taskcla, acc, lss, output_path=args.checkpoint, run_id=run_id)

    return avg_acc, gem_bwt



#######################################################################################################################


def main(args):
    utils.make_directories(args)
    utils.some_sanity_checks(args)
    utils.save_code(args)

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)


    accuracies, forgetting = [], []
    for n in range(args.num_runs):
        args.seed = n
        args.output = '{}_{}_tasks_seed_{}.txt'.format(args.experiment, args.ntasks, args.seed)
        print("args.checkpoint: {}".format(args.checkpoint))
        print("args.output: {}".format(args.output))

        print (" >>>> Run #", n)
        acc, bwt = run(args, n)
        accuracies.append(acc)
        forgetting.append(bwt)


    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))


    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()


#######################################################################################################################

if __name__ == '__main__':
    main(args)
