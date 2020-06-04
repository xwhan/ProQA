#!/bin/bash
 CUDA_VISIBLE_DEVICES=3 python3 get_embed.py \
    --do_predict \
    --prefix eval-para \
    --predict_batch_size 300 \
    --bert_model_name bert-base-uncased \
    --fp16 \
    --predict_file ../data/wiki_splits.txt \
    --init_checkpoint logs/retrieve_train.txt-seed87-bsz640-fp16True-retriever_pretraining_single-lr1e-05-bert-base-uncased-filterTrue/checkpoint_best.pt \
    --embed_save_path encodings/para_embed.npy \
    --eval-workers 32 \

