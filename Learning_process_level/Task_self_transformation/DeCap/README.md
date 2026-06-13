# Diversity-Enhanced and Classification-Aware Prompt Learning for Few-Shot Learning via Stable Diffusion

This repository contains the implementation of  
**Diversity-Enhanced and Classification-Aware Prompt Learning for Few-Shot Learning via Stable Diffusion**.  

---

## Installation
```sh
conda create -n DeCap python=3.10
conda activate DeCap
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
##  Dataset Construction

To allow flexible customization and avoid class-order bias in classification, please organize datasets in the following format:

```
dataset/
└── cifar10/
    └── train/
        ├── {class_name1,e.g. dog}/
        │   ├── img1.png
        │   └── ...
        ├── {class_name2,e.g. cat}/
        │   ├── img1.png
        │   └── ...
        └── ...
    └── val/
    └── test/
```

We only use "val" set to generate image caption prompt and to train DeCap.

---

##  Training

1. **Prepare prompt pool**

```bash
bash prepare/prepare.sh
```
2. **Generate full pool**

To accelerate training, we need to first generate synthetic dataset about the full prompt pool.

```bash
python generate.py full cifar10 10 10 None
```
3. **Start training**

```bash
python GA_train.py  \
      --per_gpu_popsize=10 \
      --prompt_nums_per_class=20 \
      --dataset=cifar10 \
      --test_annotation_path="./dataset/cifar10/cls_test.txt"  \
      --val_annotation_path="./dataset/cifar10/cls_val.txt" \
      --label_path="./dataset/cifar10/cls_classes.txt"  \
      --root_dir="./syn_dataset/cifar10/full" \
      --save_dir="./result"  \
      --pool_path="./prompt_pool/cifar10_prompt_pool.pkl"
```

---

##  Dataset Generation

After training, generate a dataset with:

```bash
python generate.py ours cifar10 20 10 /path/to/prompt_dict.pkl
```
You can cite this paper by:
```
@inproceedings{chang2025decap,
	title={Diversity-Enhanced and Classification-Aware Prompt Learning for Few-Shot Learning via Stable Diffusion},
	author={Chang, Gaoqin and Shu, Jun and Yuan, Xiang and Meng, Deyu},
	booktitle={TMLR},
	year=2025}
```
---

