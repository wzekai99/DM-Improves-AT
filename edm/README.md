# Data Generation

The generation codes are modifed based on the [official implementation of EDM](https://github.com/NVlabs/edm) and [official implementation](https://github.com/yaircarmon/semisup-adv) of [Carmon et al., 2019](https://arxiv.org/abs/1905.13736). We employ the class-conditional EDM in this implementation. 


## Requirements

- This project is tested with Ubuntu 20.04.3. 
- 4 NVIDIA A100 SXM4 40GB GPUs for training and image generation. 
- 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
- Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n edm`
  - `conda activate edm`
- For 1M data generation, we use the [official implementation](https://github.com/yaircarmon/semisup-adv) of [Carmon et al., 2019](https://arxiv.org/abs/1905.13736) to train WRN-28-10 models to give pseudo-labels, following [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946). Download selection models to `./selection_model`. 

| dataset | clean | link |
|---|:---:|---|
| CIFAR-10 | 96.15% | [cifar10_pseudo.pt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/cifar10_pseudo.pt) |
| CIFAR-100 | 80.47% | [cifar100_pseudo.pt](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/cifar100_pseudo.pt) |


## Generating data for CIFAR-10

For CIFAR-10, we generate images using the pre-trained model provided by EDM, which yields a new state-of-the-art FID of 1.79. 

For 1M data generation, following [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946), we first generate 500K images for each class and 5M in total. Generating a large number of images can be time-consuming; the workload can be distributed across multiple GPUs by launching the above command using `torchrun`:

```.bash
# Generate 500K images for each class using 4 A100 GPUs, using deterministic sampling with 20 steps
torchrun --standalone --nproc_per_node=4 generate.py --outdir=out_cifar10 --seeds=0-499999 --batch=2048  --steps=20 --class_num=10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

The name of `.npy` file indicates the label of images in the file, e.g., `1.npy`. We use the pre-trained WRN-28-10 model to score each image and select the top 100K images for each class: 

```.bash
python select_1M.py --model_path ./selection_model/cifar10_pseudo.pt --data_dir ./out_cifar10 --output_dir ./npz_cifar10 --class_num 10
```

When the amount of required generated data exceeds 1M, we merge `.npy` data files without selection. For example, generate 5M data:

```.bash
python combine_data.py --data_dir ./out_cifar10 --output_dir ./npz_cifar10 --class_num 10 --file_name 5m
```

## Generating data for CIFAR-100

For CIFAR-100, we train our own model on four A100 GPUs and select the model with the best FID (2.09) after 25 sampling steps. For 1M data generation, we first generate 50K images for each class and 5M in total: 

```.bash
# Generate 50K images for each class using 4 A100 GPUs, using deterministic sampling with 25 steps
torchrun --standalone --nproc_per_node=4 generate.py --outdir=out_cifar100 --seeds=0-49999 --batch=2048  --steps=25 --class_num=100 \
    --network=https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/edm-cifar100-cond-32x32-cond-vp.pkl
```

We use the pre-trained WRN-28-10 model to score each image and select the top 10K images for each class: 

```.bash
python select_1M.py --model_path ./selection_model/cifar100_pseudo.pt --data_dir ./out_cifar100 --output_dir ./npz_cifar100 --class_num 100
```

When the amount of required generated data exceeds 1M, we merge `.npy` data files without selection. For example, generate 5M data:

```.bash
python combine_data.py --data_dir ./out_cifar100 --output_dir ./npz_cifar100 --class_num 100 --file_name 5m
```

## License

Source code and pre-trained models of EDM are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/). [Official implementation](https://github.com/yaircarmon/semisup-adv) of [Carmon et al., 2019](https://arxiv.org/abs/1905.13736) is originally shared under the [MIT license](https://github.com/yaircarmon/semisup-adv/blob/master/LICENSE.md). 
