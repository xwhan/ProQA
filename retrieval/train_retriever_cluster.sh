#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_retriever.py \
    --do_train \
    --prefix retriever_pretraining_cluster \
    --predict_batch_size 512 \
    --bert_model_name bert-base-uncased \
    --train_batch_size 640 \
    --gradient_accumulation_steps 8 \
    --accumulate_gradients 8 \
    --efficient_eval \
    --learning_rate 1e-5 \
    --train_file ../data/data_splits/\
    --predict_file ../data/retrieve_dev_shuffled.txt \
    --seed 87 \
    --init_checkpoint logs/retrieve_train.txt-seed87-bsz640-fp16True-retriever_pretraining_single-lr1e-05-bert-base-uncased-filterTrue/checkpoint_last.pt \
    --eval-period 800 \
    --filter
