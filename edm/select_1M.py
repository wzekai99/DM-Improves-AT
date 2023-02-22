"""
Code for generating pseudolabels
"""

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import os
import argparse
import numpy as np

from torchvision import transforms
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from utils_semisup import get_model

parser = argparse.ArgumentParser(
    description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
parser.add_argument('--model_path', type=str,
                    help='path of checkpoint to standard trained model')
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--output_dir', default='data/', type=str,
                    help='directory to save output')
parser.add_argument('--top_scale', default=0.2, type=float,
                    help='scale of scoring image')
parser.add_argument('--batch_size', default=3000, type=int,
                    help='batch size of testing')
parser.add_argument('--class_num', default=10, type=int,
                    help='number of the class')
args = parser.parse_args()


if not os.path.exists(args.model_path):
    raise ValueError('Model %s not found' % args.model_path)
os.makedirs(args.output_dir, exist_ok=True)

# Loading model
checkpoint = torch.load(args.model_path)
num_classes = checkpoint.get('num_classes', args.class_num)
normalize_input = checkpoint.get('normalize_input', False)
model = get_model(args.model, 
                  num_classes=num_classes,
                  normalize_input=normalize_input)
model = nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

Fsoftmax = torch.nn.functional.softmax

final_images = []
final_targets = [] 

for class_iter in range(args.class_num):
    print(f'processing {class_iter}-th class')

    data = np.load(os.path.join(args.data_dir, str(class_iter)+'.npy'))
    unlabeled_data = CIFAR10('../cifar-data', train=False, transform=ToTensor())
    unlabeled_data.data = data
    unlabeled_data.targets = [class_iter for _ in range(unlabeled_data.data.shape[0])]
    data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            pin_memory=True)

    # Running model on unlabeled data
    confidence = []
    for i, (batch, _) in enumerate(data_loader):
        cons = Fsoftmax(model(batch.cuda()), dim=1)
        confidence.append(cons[:,class_iter].detach().cpu().numpy())

        if (i+1) % 10 == 0:
            print('Class %d Done %d/%d' % (class_iter, i+1, len(data_loader)))

    num_class_images = int(unlabeled_data.data.shape[0]*args.top_scale)
    mask = np.argsort(np.concatenate(confidence))[::-1][:num_class_images]
    final_images.append(data[mask])
    final_targets += [class_iter for _ in range(num_class_images)]

np.savez(os.path.join(args.output_dir, '1m.npz'), image=np.concatenate(final_images), label=np.array(final_targets))
