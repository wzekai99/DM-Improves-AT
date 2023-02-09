import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)
        
        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter, 
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data == 'tiny-imagenet':
            self.num_classes = 200
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups
        
        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            if self.params.consistency:
                x_aug1, x_aug2, y = x[0].to(device), x[1].to(device), y.to(device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(x_aug1, x_aug2, y, beta=self.params.beta)

            else:
                if self.params.CutMix:
                    x_all, y_all = torch.tensor([]), torch.tensor([])
                    for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                        x_all = torch.cat((x_all, x_tmp), dim=0)
                        y_all = torch.cat((y_all, y_tmp), dim=0)
                    x, y = x_all.to(device), y_all.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                
                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.beta is not None:
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)
                    
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau, 
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        update_bn(self.wa_model, self.model) 
        return dict(metrics.mean())
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix)
        return loss, batch_metrics

    def trades_loss_consistency(self, x_aug1, x_aug2, y, beta):
        """
        TRADES training with Consistency.
        """
        x = torch.cat([x_aug1, x_aug2], dim=0)
        loss, batch_metrics = trades_loss(self.model, x, y.repeat(2), self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix, use_consistency=True, cons_lambda=self.params.cons_lambda, cons_tem=self.params.cons_tem)
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """
        loss, batch_metrics = trades_loss_LSE(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          clip_value=self.params.clip_value,
                                          use_cutmix=self.params.CutMix,
                                          num_classes=self.num_classes)
        return loss, batch_metrics  

    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.wa_model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)


    def save_model_resume(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'epoch': epoch
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
    

    def load_model_resume(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
