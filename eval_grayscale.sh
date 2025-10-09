#!/bin/bash
python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/andy/lynu369/pigpigcontest/runs/domain_adapt_color/train/yolov11x_pretrain/weights/epoch50.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name 50_conf001 \
        --export_test_csv \
        --test_csv_path 50_grayscale.csv \
        --test_vis > 11x_pretrain_50_grayscale_0.01.txt     
