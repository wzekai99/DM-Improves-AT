import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base import Attack, LabelMixin

from .utils import batch_multiply
from .utils import clamp
from .utils import is_float_or_torch_tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def perturb_deepfool(xvar, yvar, predict, nb_iter=50, overshoot=0.02, ord=np.inf, clip_min=0.0, clip_max=1.0, 
                     search_iter=0, device=None):
    """
    Compute DeepFool perturbations (Moosavi-Dezfooli et al, 2016).
    Arguments:
        xvar (torch.Tensor): input images.
        yvar (torch.Tensor): predictions.
        predict (nn.Module): forward pass function.
        nb_iter (int): number of iterations.
        overshoot (float): how much to overshoot the boundary.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        search_iter (int): no of search iterations.
        device (torch.device): device to work on.
    Returns: 
        torch.Tensor containing the perturbed input, 
        torch.Tensor containing the perturbation
    """

    x_orig = xvar
    x = torch.empty_like(xvar).copy_(xvar)
    x.requires_grad_(True)
    
    batch_i = torch.arange(x.shape[0])
    r_tot = torch.zeros_like(x.data)
    for i in range(nb_iter):
        if x.grad is not None:
            x.grad.zero_()

        logits = predict(x)
        df_inds = np.argsort(logits.detach().cpu().numpy(), axis=-1)
        df_inds_other, df_inds_orig = df_inds[:, :-1], df_inds[:, -1]
        df_inds_orig = torch.from_numpy(df_inds_orig)
        df_inds_orig = df_inds_orig.to(device)
        not_done_inds = df_inds_orig == yvar
        if not_done_inds.sum() == 0:
            break

        logits[batch_i, df_inds_orig].sum().backward(retain_graph=True)
        grad_orig = x.grad.data.clone().detach()
        pert = x.data.new_ones(x.shape[0]) * np.inf
        w = torch.zeros_like(x.data)

        for inds in df_inds_other.T:
            x.grad.zero_()
            logits[batch_i, inds].sum().backward(retain_graph=True)
            grad_cur = x.grad.data.clone().detach()
            with torch.no_grad():
                w_k = grad_cur - grad_orig
                f_k = logits[batch_i, inds] - logits[batch_i, df_inds_orig]
                if ord == 2:
                    pert_k = torch.abs(f_k) / torch.norm(w_k.flatten(1), 2, -1)
                elif ord == np.inf:
                    pert_k = torch.abs(f_k) / torch.norm(w_k.flatten(1), 1, -1)
                else:
                    raise NotImplementedError("Only ord=inf and ord=2 have been implemented")
                swi = pert_k < pert
                if swi.sum() > 0:
                    pert[swi] = pert_k[swi]
                    w[swi] = w_k[swi]
        
        if ord == 2:
            r_i = (pert + 1e-6)[:, None, None, None] * w / torch.norm(w.flatten(1), 2, -1)[:, None, None, None]
        elif ord == np.inf:
            r_i = (pert + 1e-6)[:, None, None, None] * w.sign()
        
        r_tot += r_i * not_done_inds[:, None, None, None].float()
        x.data = x_orig + (1. + overshoot) * r_tot
        x.data = torch.clamp(x.data, clip_min, clip_max)
    
    x = x.detach()
    if search_iter > 0:
        dx = x - x_orig
        dx_l_low, dx_l_high = torch.zeros_like(dx), torch.ones_like(dx)
        for i in range(search_iter):
            dx_l = (dx_l_low + dx_l_high) / 2.
            dx_x = x_orig + dx_l * dx
            dx_y = predict(dx_x).argmax(-1)
            label_stay = dx_y == yvar
            label_change = dx_y != yvar
            dx_l_low[label_stay] = dx_l[label_stay]
            dx_l_high[label_change] = dx_l[label_change]
        x = dx_x
    
    # x.data = torch.clamp(x.data, clip_min, clip_max)
    r_tot = x.data - x_orig
    return x, r_tot



class DeepFoolAttack(Attack, LabelMixin):
    """
    DeepFool attack.
    [Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, 
    "DeepFool: a simple and accurate method to fool deep neural networks"]
    Arguments:
        predict (nn.Module): forward pass function.
        overshoot (float): how much to overshoot the boundary.
        nb_iter (int): number of iterations.
        search_iter (int): no of search iterations.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        ord (int): (optional) the order of maximum distortion (inf or 2).
    """
       
    def __init__(
            self, predict, overshoot=0.02, nb_iter=50, search_iter=50, clip_min=0., clip_max=1., ord=np.inf):
        super(DeepFoolAttack, self).__init__(predict, None, clip_min, clip_max)
        self.overshoot = overshoot
        self.nb_iter = nb_iter
        self.search_iter = search_iter
        self.targeted = False
        
        self.ord = ord
        assert is_float_or_torch_tensor(self.overshoot)

    def perturb(self, x, y=None):
        """
        Given examples x, returns their adversarial counterparts.
        Arguments:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted labels.
        Returns: 
            torch.Tensor containing perturbed inputs,
            torch.Tensor containing the perturbation    
        """
        
        x, y = self._verify_and_process_inputs(x, None)
        x_adv, r_adv = perturb_deepfool(x, y, self.predict, self.nb_iter, self.overshoot, ord=self.ord, 
                                        clip_min=self.clip_min, clip_max=self.clip_max, search_iter=self.search_iter, 
                                        device=device)
        return x_adv, r_adv


class LinfDeepFoolAttack(DeepFoolAttack):
    """
    DeepFool Attack with order=Linf.
    Arguments:
    Arguments:
        predict (nn.Module): forward pass function.
        overshoot (float): how much to overshoot the boundary.
        nb_iter (int): number of iterations.
        search_iter (int): no of search iterations.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    """

    def __init__(
            self, predict, overshoot=0.02, nb_iter=50, search_iter=50, clip_min=0., clip_max=1.):
        
        ord = np.inf
        super(LinfDeepFoolAttack, self).__init__(
            predict=predict, overshoot=overshoot, nb_iter=nb_iter, search_iter=search_iter, clip_min=clip_min, 
            clip_max=clip_max, ord=ord)



class L2DeepFoolAttack(DeepFoolAttack):
    """
    DeepFool Attack with order=L2.
    Arguments:
        predict (nn.Module): forward pass function.
        overshoot (float): how much to overshoot the boundary.
        nb_iter (int): number of iterations.
        search_iter (int): no of search iterations.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    """

    def __init__(
            self, predict, overshoot=0.02, nb_iter=50, search_iter=50, clip_min=0., clip_max=1.):
        
        ord = 2
        super(L2DeepFoolAttack, self).__init__(
            predict=predict, overshoot=overshoot, nb_iter=nb_iter, search_iter=search_iter, clip_min=clip_min, 
            clip_max=clip_max, ord=ord)
