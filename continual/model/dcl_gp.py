import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import quadprog
import time

from .common import MLP, ResNet18, AlexNet

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (get_model_params, BlockDecoder)

intermediate_grads = {}
def save_grad(name):
    def hook(grad):
        intermediate_grads[name] = grad
    return hook

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x)>0)

def getWeights(parameters):
    weights = np.array([])
    for param in parameters():
        weights = np.concatenate((weights,param.data.cpu().view(-1).double().numpy()),axis=0)
    return weights

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            if args.backend_net.startswith('resnet'):
                self.net = ResNet18(n_outputs)
            elif args.backend_net.startswith('alexnet'):
                self.net = AlexNet(inputsize=32, num_channels=3, num_classes=n_outputs)
            elif args.backend_net.startswith('efficientnet'):
                self.net = EfficientNet.from_pretrained(args.backend_net, num_classes=n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        self.memory_w = None
        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        self.reset_intvl = args.reset_interval
        self.reset_count = 0
        self.ref_offset = args.dcl_offset
        self.ref_count = 0
        self.knlg_decay = args.knlg_decay
        self.knlg = torch.Tensor(sum(self.grad_dims))
        if args.cuda:
            self.knlg = self.knlg.cuda()

    def forward(self, x, t):
        if 'EfficientNet' in type(self.net).__name__:
            bsz = x.size(0)
            x = x.view(bsz, 3, 32, 32)
            x = F.interpolate(x,size=(64,64), mode='bilinear', align_corners=True)
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y, x_extra, grad_learner, opt_gl):
        accm_grads = None
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            self.ref_count = 0
            self.reset_count = 0
            if self.memory_w is not None:
                accm_grads = self.memory_w.clone() 
            self.reset()

        if self.reset_intvl > 0 and self.reset_count % self.reset_intvl == self.ref_offset:
            if self.memory_w is not None:
                accm_grads = self.memory_w.clone() 
            self.reset()
        self.reset_count += 1


        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        Variable(self.memory_data[past_task]),
                        past_task)[:, offset1: offset2],
                    Variable(self.memory_labs[past_task] - offset1))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()
        
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        outputs = self.forward(x, t)[:, offset1: offset2]
        outputs.register_hook(save_grad('outputs'))
        loss = self.ce(outputs, y - offset1)
        loss.backward()

        #--------------------------
        # gradient learning begins
        #--------------------------
        loss_in_grad = torch.tensor(0)
        time_grle = torch.tensor(0)
        time_grpr = torch.tensor(0)

        start_grle = time.time()
        outputs_copy = outputs.clone().detach()
        pred_grad = grad_learner(outputs_copy.data)
        cur_lr = self.opt.param_groups[0]['lr']
        nml_pred_grad = grad_learner.gl_scale*pred_grad / \
                    torch.norm(pred_grad, dim=1, keepdim=True) * \
                    torch.norm(intermediate_grads['outputs'], dim=1, keepdim=True).mean(dim=0, keepdim=True)
        loss_in_grad = self.ce(outputs_copy-cur_lr*nml_pred_grad, y - offset1)
        opt_gl.zero_grad()
        loss_in_grad = grad_learner.gl_loss_scale*loss_in_grad
        loss_in_grad.backward()
        opt_gl.step()
        time_grle = time.time() - start_grle
        #--------------------------
        # gradient learning ends
        #--------------------------

        #--------------------------
        # gradient prediction begins
        #--------------------------
        if x_extra is not None:
            output_grad = intermediate_grads['outputs']
            start_grpr = time.time()
            extra_outputs = self.forward(x_extra, t)[:, offset1: offset2]
            with torch.no_grad():
                pred_extra_grad = grad_learner(extra_outputs)
            mean_pred_extra_grad = pred_extra_grad.data.mean(dim=0, keepdim=True)
            grad_for_update = grad_learner.gl_scale*pred_extra_grad / \
                        pred_extra_grad.norm(dim=1, keepdim=True) * \
                        output_grad.norm(dim=1, keepdim=True).mean(dim=0, keepdim=True)
            extra_outputs.backward(grad_for_update)
            time_grpr = time.time() - start_grpr
        #--------------------------
        # gradient prediction ends
        #--------------------------

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            for tt in range(t):
                if self.grads[:, tt].abs().sum() != 0:
                    tmp = torch.dot(self.grads[:, tt],self.grads[:, t]) / (torch.norm(self.grads[:, tt])*torch.norm(self.grads[:, t]))

            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            ref_diretion = None
            if self.memory_w is not None:
                ref_diretion = torch.from_numpy(getWeights(self.parameters)) - self.memory_w
                ref_diretion = ref_diretion.unsqueeze(1)
            
            mem_grads = None
            if ref_diretion is None or ref_diretion.sum()==0:
                mem_grads = self.grads.index_select(1, indx)
            else:
                mem_grads = torch.cat((self.grads.index_select(1, indx),ref_diretion.type_as(self.grads)), dim=1)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            mem_grads)
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              mem_grads, self.margin)

                tmp_cos = 0
                for tt in range(t):
                    if self.grads[:, tt].abs().sum() != 0:
                        tmp = torch.dot(self.grads[:, tt],self.grads[:, t]) / (torch.norm(self.grads[:, tt])*torch.norm(self.grads[:, t]))
                        tmp_cos += tmp.item()
                tmp_cos /= t
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

        self.opt.step()

        #if accm_grads is not None:
        if self.memory_w is None:
            self.memory_w = torch.from_numpy(getWeights(self.parameters))

        return

    def reset(self):
        self.memory_w = None
