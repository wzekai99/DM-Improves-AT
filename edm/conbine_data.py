"""
Code for generating pseudolabels
"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--output_dir', default='data/', type=str,
                    help='directory to save output')
parser.add_argument('--class_num', default=10, type=int,
                    help='number of the class')
parser.add_argument('--file_name', default='5m', type=str,
                    help='name of output file')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

final_images = []
final_targets = [] 

for class_iter in range(args.class_num):
    data = np.load(os.path.join(args.data_dir, str(class_iter)+'.npy'))

    num_class_images = int(data.shape[0])
    final_images.append(data)
    final_targets += [class_iter for _ in range(num_class_images)]

np.savez(os.path.join(args.output_dir, args.file_name+'.npz'), image=np.concatenate(final_images), label=np.array(final_targets))
