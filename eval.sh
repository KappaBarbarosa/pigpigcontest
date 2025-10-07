#!/bin/bash
# python train_domain_adaptation.py --eval_only --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_color/yolov10x_pretrain/weights/epoch30.pt  --color_mode color --eval_test_sample > eval_color.txt

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/epoch10.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name epoch10 \
        --test_vis > eval_grayscale_epoch10.txt     


python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/epoch20.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name epoch20 \
        --test_vis > eval_grayscale_epoch20.txt     

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/epoch30.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name epoch30 \
        --test_vis > eval_grayscale_epoch30.txt     

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/best.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name best \
        --test_vis > eval_grayscale_best.txt     

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_pretrain/weights/last.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name last \
        --test_vis > eval_grayscale_last.txt     

python train_domain_adaptation.py --eval_only  \
        --checkpoint /home/marl2025/pigpigcontest/runs/domain_adapt_grayscale/yolov10x_finetune/weights/best.pt  \
        --eval_test_sample --color_mode grayscale  \
        --name finetune \
        --test_vis > eval_grayscale_finetune.txt     
