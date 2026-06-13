#!/usr/bin/env bash

dataset="cifar10"

CUDA_VISIBLE_DEVICES=0 python prepare/label.py $dataset

CUDA_VISIBLE_DEVICES=0 python prepare/txt_annotation.py $dataset

CUDA_VISIBLE_DEVICES=0 python prepare/LE.py 100 $dataset

CUDA_VISIBLE_DEVICES=0 python prepare/BLIP_generate.py $dataset

CUDA_VISIBLE_DEVICES=0 python prepare/pool.py $dataset

CUDA_VISIBLE_DEVICES=0 python prepare/get_zeroshot.py $dataset
