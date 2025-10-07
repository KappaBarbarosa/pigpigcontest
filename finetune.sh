#!/bin/bash
# python train_domain_adaptation.py --skip_pretrain --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_color/yolov10x_pretrain/weights/epoch30.pt  --color_mode color --eval_test_sample --device cuda:0
python train_domain_adaptation.py --skip_pretrain --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/epoch30.pt  --color_mode grayscale --eval_test_sample --device cuda:0

# python train_domain_adaptation.py --eval_only --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/epoch30.pt  --eval_test_sample --color_mode grayscale > eval_grayscale.txt
