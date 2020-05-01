#!/bin/bash


# # nq train
# CUDA_VISIBLE_DEVICES=0 python train_retrieve_qa.py \
# --do_train \
# --prefix openqa-dense-index-nq \
# --eval_period 1000 \
# --bert_model_name bert-base-uncased \
# --train_batch_size 40 \
# --gradient_accumulation_steps 8 \
# --accumulate_gradients 8 \
# --efficient_eval \
# --learning_rate 1e-5 \
# --fp16 \
# --raw-train-data /home/xwhan/code/DrQA/data/datasets/nq-train.txt \
# --raw-eval-data /home/xwhan/code/DrQA/data/datasets/nq-dev-2000.txt \
# --seed 42 \
# --retriever-path retrieval/logs/final_splits_spherical-seed19940817-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterFalse/checkpoint_100k.pt \
# --index-path retrieval/index_data/para_embed_100k.npy \
# --fix-para-encoder \
# --matched-para-path /home/xwhan/retrieval_data/nq_ft_train_matched_10000.txt \
# --shared-norm \
# --separate
# --add-select \



# CUDA_VISIBLE_DEVICES=2 python train_retrieve_qa.py \
# --do_train \
# --prefix dense-index-wq-combined \
# --eval_period -1 \
# --bert_model_name bert-base-uncased \
# --train_batch_size 20 \
# --gradient_accumulation_steps 4 \
# --accumulate_gradients 4 \
# --efficient_eval \
# --learning_rate 1e-5 \
# --fp16 \
# --raw-train-data /home/xwhan/code/DrQA/data/datasets/wq-train-combined.txt \
# --raw-eval-data /home/xwhan/code/DrQA/data/datasets/wq-test.txt \
# --seed 42 \
# --retriever-path retrieval/logs/final_splits_spherical-seed19940817-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterFalse/checkpoint_100k.pt \
# --index-path retrieval/index_data/para_embed_100k.npy \
# --fix-para-encoder \
# --matched-para-path /home/xwhan/retrieval_data/wq_ft_train-combined_matched_15000.txt \
# --shared-norm \
# --add-select \
# --no-joint \
# --use-spanbert \
# --qa-drop 0.05



## trec




# CUDA_VISIBLE_DEVICES=0 python train_retrieve_qa.py \
# --do_train \
# --prefix dense-index-trec \
# --eval_period -1 \
# --bert_model_name bert-base-uncased \
# --train_batch_size 5 \
# --gradient_accumulation_steps 1 \
# --accumulate_gradients 1 \
# --efficient_eval \
# --learning_rate 1e-5 \
# --fp16 \
# --raw-train-data /home/xwhan/code/DrQA/data/datasets/trec-train.txt \
# --raw-eval-data /home/xwhan/code/DrQA/data/datasets/trec-dev.txt \
# --seed 3 \
# --retriever-path retrieval/logs/final_splits_spherical-seed19940817-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterFalse/checkpoint_100k.pt \
# --index-path retrieval/index_data/para_embed_100k.npy \
# --fix-para-encoder \
# --num_train_epochs 10 \
# --matched-para-path /home/xwhan/retrieval_data/trec_train_matched_20000.txt \
# --regex \
# --shared-norm \
# --separate \





# check whether no-cluser is better than ORQA

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
--raw-train-data /home/xwhan/code/DrQA/data/datasets/trec-train.txt \
--raw-eval-data /home/xwhan/code/DrQA/data/datasets/trec-dev.txt \
--seed 3 \
--retriever-path  retrieval/logs/retrieve_train.txt-seed31-bsz640-fp16True-baseline_no_cluster_from_failed_continue-lr1e-05-bert-base-uncased-filterTrue/checkpoint_40000.pt \
--index-path retrieval/index_data/para_embed_ablation_40k.npy \
--fix-para-encoder \
--num_train_epochs 10 \
--matched-para-path /home/xwhan/retrieval_data/trec_train_matched_20000.txt \
--regex \
--shared-norm \
# --separate \
