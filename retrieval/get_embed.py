import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from copy import deepcopy

from torch.utils.data import DataLoader
from datasets import EmDataset, em_collate
from retriever import BertForRetriever
from transformers import AdamW, BertConfig, BertTokenizer
from utils import move_to_cuda, convert_to_half, AverageMeter
from config import get_args

from collections import defaultdict, namedtuple
import torch.nn.functional as F


def load_saved(model, path):
    state_dict = torch.load(path)
    def filter(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def main():
    args = get_args()

    is_query_embed = args.is_query_embed
    embed_save_path = args.embed_save_path

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError(
                "If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = BertConfig.from_pretrained(args.bert_model_name)
    model = BertForRetriever(bert_config, args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    if args.do_train and args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    eval_dataset = EmDataset(
        tokenizer, args.predict_file, args.max_query_length, args.max_seq_length, is_query_embed)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=em_collate, pin_memory=True, num_workers=args.eval_workers)

    assert args.init_checkpoint != ""
    model = load_saved(model, args.init_checkpoint)

    model.to(device)

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    embeds = predict(args, model, eval_dataloader, device, fp16=args.efficient_eval, is_query_embed=is_query_embed)
    np.save(embed_save_path, embeds.cpu().numpy())


def predict(args, model, eval_dataloader, device, fp16=False, is_query_embed=True):
    if type(model) == list:
        model = [m.eval() for m in model]
    else:
        model.eval()
    if fp16:
        if type(model) == list:
            model = [m.half() for m in model]
        else:
            model.half()

    num_correct = 0.0
    num_total = 0.0
    embed_array = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            results = model.get_embed(batch_to_feed, is_query_embed)
            embed = results['embed']
            embed_array.append(embed)
            #print(prediction, target, sum(prediction==target), len(prediction))
            #print(num_total, num_correct)

    ## linear combination tuning on dev data
    embed_array = torch.cat(embed_array)

    if fp16:
        model.float()

    model.train()
    return embed_array


if __name__ == "__main__":
    main()
