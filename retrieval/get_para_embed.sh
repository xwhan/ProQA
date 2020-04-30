#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python3 get_embed.py \
#     --prefix c10000_encoder \
#     --do_predict \
#     --prefix eval-para \
#     --predict_batch_size 2048 \
#     --bert_model_name bert-base-uncased \
#     --efficient_eval \
#     --predict_file /data/hongwang/nq_rewrite/db/para_doc.db \
#     --init_checkpoint logs/splits_3_28_c10000-seed42-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterTrue/checkpoint_best.pt \
#     --embed_save_path index_data/para_embed_3_28_c10000.npy \
#     --eval-workers 32 \
 

#  # Encode the whole training data for clustering

 CUDA_VISIBLE_DEVICES=3 python3 ../get_embed.py \
    --do_predict \
    --prefix eval-para \
    --predict_batch_size 2048 \
    --bert_model_name bert-base-uncased \
    --efficient_eval \
    --predict_file /data/hongwang/nq_rewrite/db/para_doc.db \
    --init_checkpoint logs/retrieve_train.txt-seed31-bsz640-fp16True-baseline_no_cluster_from_failed_continue-lr1e-05-bert-base-uncased-filterTrue/checkpoint_best.pt \
    --embed_save_path index_data/para_embed_ablation_80k.npy \
    --eval-workers 32 \
 