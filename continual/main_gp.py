import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import math

import numpy as np
import scipy.io

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data as data
from torch.autograd import Variable
from metrics.metrics import confusion_matrix

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

from datasets import dataset

# continuum iterator #########################################################

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.std_mem = np.array([])

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std_mem = np.append(self.std_mem, val)
        self.std = np.std(self.std_mem)

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
    #return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        for t in range(n_tasks):
            N = data[t][1].size(0)
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]

class RegressionNet(nn.Module):
    def __init__(self, dim, gal_arch=[64],
            gl_scale=0.01, gl_loss_scale=1.0):
        super().__init__()
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
                net.append(nn.Linear(in_channel, gal_arch[i], bias=False))
                net.append(nn.ReLU(inplace=True))
                in_channel = gal_arch[i]
            net.append(nn.Linear(in_channel, self.dim, bias=False))

            self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x

# train handle ###############################################################


def eval_tasks(model, tasks, args):
    model.eval()
    result = []
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0
        
        eval_bs = x.size(0)

        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                xb = x[b_from].view(1, -1)
                yb = torch.LongTensor([y[b_to]]).view(1, -1)
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]
            if args.cuda:
                xb = xb.cuda()
            #xb = Variable(xb, volatile=True)
            with torch.no_grad():
                xb = Variable(xb)
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            rt += (pb == yb).float().sum()

        result.append(rt / x.size(0))

    return result

def life_experience(model, continuum, x_te, args, extradata):
    result_a = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        if(((i % args.log_every) == 0) or (t != current_task)):
            result_a.append(eval_tasks(model, x_te, args))
            result_t.append(current_task)
            current_task = t

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(Variable(v_x), t, Variable(v_y))

    result_a.append(eval_tasks(model, x_te, args))
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), time_spent

def life_experience_gl(model, continuum, x_te, args, \
                        extradata, grad_learner, opt_gl):
    result_a = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        if(((i % args.log_every) == 0) or (t != current_task)):
            result_a.append(eval_tasks(model, x_te, args))
            result_t.append(current_task)
            current_task = t

        x_extra = None
        if i >= args.gl_start_predict and random.uniform(0, 1) < args.gl_prob:
            if args.extra_data_name == 'cub':
                x_extra, _, _, _ = next(extradata)
            else:
                x_extra, _ = next(extradata)
            x_extra = x_extra.squeeze().view(x_extra.size(0),-1)
            if args.cuda:            
                x_extra = x_extra.cuda()
            x_extra = Variable(x_extra)

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        if 'model.gem' in type(model).__module__ or 'model.dcl' in type(model).__module__:
            model.observe(Variable(v_x), t, Variable(v_y), x_extra, grad_learner, opt_gl)
        else:
            model.observe(Variable(v_x), t, Variable(v_y), Variable(x_extra))

    result_a.append(eval_tasks(model, x_te, args))
    result_t.append(current_task)
    
    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), time_spent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--reset_interval', type=int, default=200,
                        help='dcl reset interval')
    parser.add_argument('--backend_net', type=str, default='resnet',
                        help='Backend network: alexnet, resnet, efficientnet-b1')
    parser.add_argument('--dcl_offset', type=int, default=0,
                        help='dcl offset')
    parser.add_argument('--dcl_refnum', type=int, default=1,
                        help='dcl reference number')
    parser.add_argument('--knlg_decay', type=float, default=0.0,
                        help='knowledge decay')

    parser.add_argument('--extra-input-size', type=int, default=32, help='lr scale used in GAL')
    parser.add_argument('--extra-data-name', type=str, default='', help='the path to the datasets')
    parser.add_argument('--extra-data', type=str, default='', help='the path to the datasets')
    parser.add_argument('--extra-batch', default=128, type=int, metavar='N',
                        help='extra batchsize')
    parser.add_argument('--gl_arch', nargs='+', type=int, help='arch of gradient learner')
    parser.add_argument('--gl_loss_scale', type=float, default=1.0, help='loss scale used in gradient learning')    
    parser.add_argument('--gl_scale', type=float, default=1, help='lr scale used in GAL')
    parser.add_argument('--gl_dim', type=int, default=10, help='the dimension of output gradient')
    parser.add_argument('--gl_prob', type=float, default=2, help='probability threshold')
    parser.add_argument('--gl_start_predict', type=int, default=0, help='postponed learning steps for gradient prediction')

    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False

    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)

    x_in_size = args.extra_input_size
    extradata = None
    if 'tiny4mnist' in args.extra_data_name:
        extra_data_dir = os.path.join(args.extra_data, 'train')
        normalize = transforms.Normalize(mean=[0.456, 0.406],
                                         std=[0.224, 0.225])
        extra_data_loader = data.DataLoader(
            datasets.ImageFolder(extra_data_dir, transforms.Compose([
                transforms.RandomCrop((60,60)),
                transforms.Resize(x_in_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])),
            batch_size=args.extra_batch, shuffle=True,
            num_workers=2)
        extradata = iter(cycle(extra_data_loader))
    elif 'tiny' in args.extra_data_name:
        extra_data_dir = os.path.join(args.extra_data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        extra_data_loader = data.DataLoader(
            datasets.ImageFolder(extra_data_dir, transforms.Compose([
                transforms.RandomCrop((60,60)),
                transforms.Resize(x_in_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.extra_batch, shuffle=True,
            num_workers=2)
        extradata = iter(cycle(extra_data_loader))
    elif 'coco' in args.extra_data_name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        extra_tfms = transforms.Compose([
                 transforms.Resize((x_in_size,x_in_size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize])
        extra_dataset = ImageFolder(args.extra_data, extra_tfms)
        extra_data_loader = torch.utils.data.DataLoader(
            extra_dataset, batch_size=args.extra_batch, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False)
        extradata = iter(cycle(extra_data_loader))
    elif 'cub' in args.extra_data_name:
        extraset = dataset.CUB(input_size=x_in_size, root=args.extra_data, is_train=True)
        extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                              shuffle=True, num_workers=2, drop_last=False)
        extradata = iter(cycle(extra_data_loader))
    elif 'car' in args.extra_data_name:
        dataset_name = 'car'
        extraset = dataset.STANFORD_CAR(input_size=x_in_size, root=args.extra_data, is_train=True)
        extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                              shuffle=True, num_workers=2, drop_last=False)
        extradata = iter(cycle(extra_data_loader))
    elif 'aircraft' in args.extra_data_name:
        dataset_name = 'Aircraft'
        extraset = dataset.FGVC_aircraft(input_size=x_in_size, root=args.extra_data, is_train=True)
        extra_data_loader = data.DataLoader(extraset, batch_size=args.extra_batch,
                                              shuffle=True, num_workers=2, drop_last=False)
        extradata = iter(cycle(extra_data_loader))

    # set up continuum
    continuum = Continuum(x_tr, args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    grad_learner = RegressionNet(args.gl_dim, args.gl_arch, 
                gl_scale=args.gl_scale, gl_loss_scale=args.gl_loss_scale)
    if args.cuda:
        model.cuda()
        grad_learner.cuda()
    opt_gl = optim.SGD(grad_learner.parameters(), args.lr)

    output_str = '====>Total params: {:.2f}M + {}'.format(
            sum(p.numel() for p in model.parameters())/1000000.0,
            sum(p.numel() for p in grad_learner.parameters()))
    print(output_str)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    print(type(model).__module__)
    if 'gem' in type(model).__module__ or 'dcl' in type(model).__module__:
        # run model on continuum
        result_t, result_a, spent_time = life_experience_gl(
            model, continuum, x_te, args, extradata, grad_learner, opt_gl)
    else:
        result_t, result_a, spent_time = life_experience(
            model, continuum, x_te, args, extradata)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.4f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
