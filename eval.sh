#!/bin/bash
# python train_domain_adaptation.py --eval_only --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_color/yolov10x_pretrain/weights/epoch30.pt  --color_mode color --eval_test_sample > eval_color.txt

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/None/yolov11x_pretrain/weights/last.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name last \
        --export_test_csv \
        --test_csv_path last_grayscale.csv \
        --test_vis > 11x_pretrain_last_0.1.txt     


python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/None/yolov11x_pretrain/weights/best.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name best \
        --export_test_csv \
        --test_csv_path best_grayscale.csv \
        --test_vis > 11x_pretrain_best_0.1.txt   
