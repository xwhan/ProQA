#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,3,4 python train_retriever.py \
#     --do_train \
#     --prefix retrieve-from94_c1000_continue_from_failed \
#     --predict_batch_size 512 \
#     --bert_model_name bert-base-uncased \
#     --train_batch_size 640 \
#     --gradient_accumulation_steps 8 \
#     --accumulate_gradients 8 \
#     --efficient_eval \
#     --learning_rate 1e-5 \
#     --fp16 \
#     --train_file /home/xwhan/retrieval_data/final_splits_spherical \
#     --predict_file /data/hongwang/nq_rewrite/db/re_data/retrieve_dev_shuffled.txt \
#     --seed 19940817 \
#     --init_checkpoint logs/splits_3_28_c10000-seed42-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterTrue/checkpoint_best.pt \
#     --eval-period 800 \
    # --filter


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_retriever.py \
    --do_train \
    --prefix baseline_no_cluster_from_failed_again_continue \
    --predict_batch_size 512 \
    --bert_model_name bert-base-uncased \
    --train_batch_size 640 \
    --gradient_accumulation_steps 8 \
    --accumulate_gradients 8 \
    --efficient_eval \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file /data/hongwang/nq_rewrite/db/re_data/retrieve_train.txt \
    --predict_file /data/hongwang/nq_rewrite/db/re_data/retrieve_dev_shuffled.txt \
    --seed 87 \
    --init_checkpoint logs/retrieve_train.txt-seed31-bsz640-fp16True-baseline_no_cluster_from_failed_continue-lr1e-05-bert-base-uncased-filterTrue/checkpoint_last.pt \
    --eval-period 800 \
    --filter
