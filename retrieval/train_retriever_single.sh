#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_retriever.py \
    --do_train \
    --prefix retriever_pretraining_single \
    --predict_batch_size 512 \
    --bert_model_name bert-base-uncased \
    --train_batch_size 640 \
    --gradient_accumulation_steps 8 \
    --accumulate_gradients 8 \
    --efficient_eval \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file ../data/retrieve_train.txt \
    --predict_file ../data/retrieve_dev_shuffled.txt \
    --seed 87 \
    --eval-period 800 \
    --filter
