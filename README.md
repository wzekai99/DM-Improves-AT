# Better Diffusion Models Further Improve Adversarial Training

Code for the paper [Better Diffusion Models Further Improve Adversarial Training](https://arxiv.org/pdf/2302.04638.pdf) (ICML 2023).



## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 20.04.3
- GPU: NVIDIA A100
- Cuda: 11.1, Cudnn: v8.2
- Python: 3.9.5
- PyTorch: 1.8.0
- Torchvision: 0.9.0

## Acknowledgement
The adversarial training codes are modifed based on the [PyTorch implementation](https://github.com/imrahulr/adversarial_robustness_pytorch) of [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946). The generation codes are modifed based on the [official implementation of EDM](https://github.com/NVlabs/edm). For data generation, please refer to [`edm/README.md`](./edm) for more details.  

## Requirements

- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```.bash
pip install git+https://github.com/fra31/auto-attack
```

- Install or download [RandAugment](https://github.com/ildoonet/pytorch-randaugment):
```.bash
pip install git+https://github.com/ildoonet/pytorch-randaugment
```

- Download EDM generated data to `./edm_data/cifar10` and `./edm_data/cifar100`. Since 20M and 50M data files are too large, we split them into several parts:

| dataset | size | link |
|---|:---:|:---:|
| CIFAR-10 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz) |
| CIFAR-10 | 5M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/5m.npz) |
| CIFAR-10 | 10M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/10m.npz) |
| CIFAR-10 | 20M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part2.npz) |
| CIFAR-10 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part4.npz) |
| CIFAR-100 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz) |
| CIFAR-100 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part4.npz) |

- Merge 20M and 50M generated data: 
  
```
python merge-data.py
```

## Training Commands

Run [`train-wa.py`](./train-wa.py) for reproducing the results reported in the papers. For example, train a WideResNet-28-10 model via [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10 with the 1M additional generated data provided by EDM ([Karras et al., 2022](https://github.com/NVlabs/edm)):

```.bash
python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'edm_data/cifar10/1m.npz' \
    --ls 0.1
```

## Evaluation Commands
The trained models can be evaluated by running [`eval-aa.py`](./eval-aa.py) which uses [AutoAttack](https://github.com/fra31/auto-attack) for evaluating the robust accuracy. Run the command (taking the checkpoint above as an example):

```.bash
python eval-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1'
```

To evaluate the model on last epoch under AutoAttack, run the command: 

```.bash
python eval-last-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1'
```


## Pre-trained checkpoints


We provide the state-of-the-art pre-trained checkpoints of WRN-28-10 (Swish) and WRN-70-16 (Swish). Clean and robust accuracies are measured on the full test set. The robust accuracy is measured using [AutoAttack](https://github.com/fra31/auto-attack).

| dataset | norm | radius | architecture | clean | robust | link |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | 92.44% | 67.31% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_linf_wrn28-10.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar10_linf_wrn28-10.txt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | 93.25% | 70.69% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_linf_wrn70-16.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar10_linf_wrn70-16.txt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-28-10 | 95.16% | 83.63% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_l2_wrn28-10.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar10_l2_wrn28-10.txt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-70-16 | 95.54% | 84.86% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_l2_wrn70-16.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar10_l2_wrn70-16.txt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | 72.58% | 38.83% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar100_linf_wrn28-10.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar100_linf_wrn28-10.txt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | 75.22% | 42.67% | [checkpoint](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar100_linf_wrn70-16.pt) [argtxt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/argtxt/cifar100_linf_wrn70-16.txt)

- **Downloading `checkpoint` to `trained_models/mymodel/weights-best.pt`**
- **Downloading `argtxt` to `trained_models/mymodel/args.txt`**
  
For evaluation under AutoAttack, run the command:

```.bash
python eval-aa.py --data-dir 'cifar-data' --log-dir 'trained_models' --desc 'mymodel'
```

## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{wang2023better,
  title={Better Diffusion Models Further Improve Adversarial Training},
  author={Wang, Zekai and Pang, Tianyu and Du, Chao and Lin, Min and Liu, Weiwei and Yan, Shuicheng},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

and/or our related works
```bib
@inproceedings{pang2022robustness,
  title={Robustness and Accuracy Could be Reconcilable by (Proper) Definition},
  author={Pang, Tianyu and Lin, Min and Yang, Xiao and Zhu, Jun and Yan, Shuicheng},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```
```bib
@inproceedings{pang2021bag,
  title={Bag of Tricks for Adversarial Training},
  author={Pang, Tianyu and Yang, Xiao and Dong, Yinpeng and Su, Hang and Zhu, Jun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
