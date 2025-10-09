#!/bin/bash
python train_domain_adaptation.py \
    --color_mode color \
    --model yolov11x \
    --device cuda:0 \
    --pretrain_epochs 100 \
    --batch 8 \
    --train_all