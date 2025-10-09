#!/bin/bash
python train_domain_adaptation.py \
    --color_mode grayscale \
    --model yolov11x \
    --device cuda:0 \
    --pretrain_epochs 100 \
    --batch 8 \
    --train_all