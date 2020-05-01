#!/bin/bash
mkdir retriever_data
 CUDA_VISIBLE_DEVICES=3 python3 get_embed.py \
    --do_predict \
    --prefix eval-para \
    --predict_batch_size 2048 \
    --bert_model_name bert-base-uncased \
    --efficient_eval \
    --predict_file ../data/para_doc.db \
    --init_checkpoint logs/retrieve_train.txt-seed87-bsz640-fp16True-retriever_pretraining_single-lr1e-05-bert-base-uncased-filterTrue/checkpoint_last.pt \
    --embed_save_path retriever_data/para_embed.npy \
    --eval-workers 32 \

