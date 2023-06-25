import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .ti_wideresnetwithswish import ti_wideresnetwithswish

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish']


def create_model(name, normalize, info, device):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if info['data'] in ['tiny-imagenet', 'tiny-imagenets']:
        if 'wrn' in name and 'swish' in name:
            backbone = ti_wideresnetwithswish(name, num_classes=info['num_classes'], device=device)
        else:
            from .ti_preact_resnet import ti_preact_resnet
            backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)
    
    elif info['data'] in DATASETS and info['data'] not in ['tiny-imagenet', 'tiny-imagenets']:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
