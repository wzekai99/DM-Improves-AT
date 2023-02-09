# Better Diffusion Models Further Improve Adversarial Training



## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 20.04.3
- GPU: NVIDIA A100
- Cuda: 11.1, Cudnn: v8.2
- Python: 3.9.5
- PyTorch: 1.8.0
- Torchvision: 0.9.0

## Acknowledgement
The codes are modifed based on the [PyTorch implementation](https://github.com/imrahulr/adversarial_robustness_pytorch) of [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946).

## Requirements

- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```
pip install git+https://github.com/fra31/auto-attack
```

- Install or download [RandAugment](https://github.com/ildoonet/pytorch-randaugment):
```
pip install git+https://github.com/ildoonet/pytorch-randaugment
```

- Download EDM generated data to `edm_data`. Since 20M and 50M data files are too large, we split them into several parts:

| dataset | size | link |
|---|:---:|:---:|
| CIFAR-10 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz) |
| CIFAR-10 | 5M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/5m.npz) |
| CIFAR-10 | 10M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/10m_ran0.npz) |
| CIFAR-10 | 20M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part2.npz) |
| CIFAR-10 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part4.npz) |
| CIFAR-100 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz) |
| CIFAR-100 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part4.npz) |

- Merge 20M and 50M generated data: 
  
```
python merge_data.py
```

## Training Commands

Run [`train-wa.py`](./train-wa.py) for reproducing the results reported in the papers. For example, train a WideResNet-28-10 model via [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10 with the 1M additional generated data provided by EDM ([Karras et al., 2022](https://github.com/NVlabs/edm)):

```python
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
    --aux-data-filename 'edm_data/1m.npz' \
    --ls 0.1
```



## Downloading models

We provide checkpoints which  Download a model from links listed in the following table. Clean and robust accuracies are measured on the full test set. The robust accuracy is measured using [AutoAttack](https://github.com/fra31/auto-attack).

| dataset | size | link |
|---|:---:|:---:|
| CIFAR-10 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz) |
| CIFAR-10 | 5M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/5m.npz) |
| CIFAR-10 | 10M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/10m_ran0.npz) |
| CIFAR-10 | 20M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part2.npz) |
| CIFAR-10 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/50m_part4.npz) |
| CIFAR-100 | 1M | [npz](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz) |
| CIFAR-100 | 50M | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part2.npz) [part3](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part3.npz) [part4](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/50m_part4.npz) |

- **Downloading `checkpoint` to `trained_models/mymodel/weights-best.pt`**
- **Downloading `argtxt` to `trained_models/mymodel/args.txt`**

## Evaluation Commands
The trained models can be evaluated by running [`eval-aa.py`](./eval-aa.py) which uses [AutoAttack](https://github.com/fra31/auto-attack) for evaluating the robust accuracy. Run the command:

```python
python eval-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc mymodel
```

To evaluate the model on last epoch under AutoAttack, run the command: 

```python
python eval-last-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc <path to checkpoint of last epoch>
```
