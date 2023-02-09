import numpy as np

import torch
from autoattack.autopgd_base import APGDAttack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class APGD():
    """
    APGD attack (from AutoAttack) (Croce et al, 2020).
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        ord (int): (optional) the order of maximum distortion (inf or 2).
    """
    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, ord=np.inf, seed=1):
        assert loss_fn in ['ce', 'dlr'], 'Only loss_fn=ce or loss_fn=dlr are supported!'
        assert ord in [2, np.inf], 'Only ord=inf or ord=2 are supported!'
        
        norm = 'Linf' if ord == np.inf else 'L2'
        self.apgd = APGDAttack(predict, n_restarts=n_restarts, n_iter=nb_iter, verbose=False, eps=eps, norm=norm, 
                               eot_iter=1, rho=.75, seed=seed, device=device)
        self.apgd.loss = loss_fn

    def perturb(self, x, y):
        x_adv = self.apgd.perturb(x, y)[1]
        r_adv = x_adv - x
        return x_adv, r_adv

    
class LinfAPGDAttack(APGD):
    """
    APGD attack (from AutoAttack) with order=Linf.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """
    
    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, seed=1):
        ord = np.inf
        super(L2APGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, n_restarts=n_restarts, eps=eps, nb_iter=nb_iter, ord=ord, seed=seed)


class L2APGDAttack(APGD):
    """
    APGD attack (from AutoAttack) with order=L2.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """
    
    def __init__(self, predict, loss_fn='ce', n_restarts=2, eps=0.3, nb_iter=40, seed=1):
        ord = 2
        super(L2APGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, n_restarts=n_restarts, eps=eps, nb_iter=nb_iter, ord=ord, seed=seed)