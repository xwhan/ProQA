#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python train_retrieve_qa.py \
--do_train \
--prefix dense-index-trec-nocluser-70k \
--eval_period -1 \
--bert_model_name bert-base-uncased \
--train_batch_size 5 \
--gradient_accumulation_steps 1 \
--accumulate_gradients 1 \
--efficient_eval \
--learning_rate 1e-5 \
--fp16 \
--raw-train-data ../data/trec-train.txt \
--raw-eval-data ../data/trec-dev.txt \
--seed 3 \
--retriever-path  ../retrieval/logs/retrieve_train.txt-seed31-bsz640-fp16True-baseline_no_cluster_from_failed_continue-lr1e-05-bert-base-uncased-filterTrue/checkpoint_40000.pt \
--index-path ../retrieval/encodings/para_embed.npy \
--fix-para-encoder \
--num_train_epochs 10 \
--matched-para-path ../data/trec_train_matched_20000.txt \
--regex \
--shared-norm \
# --separate \
