import torch
import torch.nn as nn

from .base import Attack, LabelMixin
from .utils import batch_multiply
from .utils import clamp 


class FGSMAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): attack step size.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        super(FGSMAttack, self).__init__(predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y  (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns: 
            torch.Tensor containing perturbed inputs.
            torch.Tensor containing the perturbation.
        """

        x, y = self._verify_and_process_inputs(x, y)
        
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)
        xadv = clamp(xadv, self.clip_min, self.clip_max)
        radv = xadv - x
        return xadv.detach(), radv.detach()


LinfFastGradientAttack = FGSMAttack


class FGMAttack(Attack, LabelMixin):
    """
    One step fast gradient method. Perturbs the input with gradient (not gradient sign) of the loss wrt the input.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): attack step size.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0., clip_max=1., targeted=False):
        super(FGMAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y  (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns: 
            torch.Tensor containing perturbed inputs.
            torch.Tensor containing the perturbation.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        xadv = xadv + batch_multiply(self.eps, grad)
        xadv = clamp(xadv, self.clip_min, self.clip_max)
        radv = xadv - x

        return xadv.detach(), radv.detach()


L2FastGradientAttack = FGMAttack