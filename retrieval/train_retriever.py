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
from datasets import ReDataset, ReSampler, re_collate, ClusterSampler, ClusterDataset
from retriever import BertForRetriever
from transformers import AdamW, BertConfig, BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from utils import move_to_cuda, convert_to_half, AverageMeter
from config import get_args

from collections import defaultdict, namedtuple
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def load_saved(model, path):
    state_dict = torch.load(path)
    def filter(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def main():
    args = get_args()

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # tb logger
    data_name = args.train_file.split("/")[-1].split('-')[0]
    model_name = f"{data_name}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-{args.prefix}-lr{args.learning_rate}-{args.bert_model_name}-filter{args.filter}"
    tb_logger = SummaryWriter(os.path.join(
        args.output_dir, "tflogs", model_name))
    args.output_dir = os.path.join(args.output_dir, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

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

    eval_dataset = ReDataset(
        tokenizer, args.predict_file, args.max_query_length, args.max_seq_length)
    #sampler = ReSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=re_collate, pin_memory=True, num_workers=args.eval_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        if ";" in args.init_checkpoint:
            models = []
            for path in args.init_checkpoint.split(";"):
                instance = deepcopy(load_saved(model, path))
                models.append(instance)
            model = models
        else:
            model = load_saved(model, args.init_checkpoint)

    if type(model) == list:
        model = [m.to(device) for m in model]
    else:
        model.to(device)
        print(
            f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_acc = 0
        wait_step = 0
        stop_training = False
        train_loss_meter = AverageMeter()
        model.train()

        if not os.path.isdir(args.train_file):
            train_dataset = ReDataset(
                tokenizer, args.train_file, args.max_query_length, args.max_seq_length, args.filter)
            sampler = ReSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler, pin_memory=True, collate_fn=re_collate, num_workers=8)
        else:
            train_dataset = ClusterDataset(
                tokenizer, args.train_file, args.max_query_length, args.max_seq_length, args.filter)
            sampler = ClusterSampler(
                train_dataset, args.train_batch_size)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler, pin_memory=True, collate_fn=re_collate, num_workers=8)
        
        logger.info('Start training....')
        loss_fct = CrossEntropyLoss()
        for epoch in range(int(args.num_train_epochs)):
            
            for batch in tqdm(train_dataloader):
                batch_step += 1
                batch = move_to_cuda(batch)
                outputs = model(batch)

                product = torch.mm(outputs["q"], outputs["c"].t())
                target = torch.arange(product.size(0)).to(product.device)
                loss = loss_fct(product, target)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                train_loss_meter.update(loss.item())
                tb_logger.add_scalar('batch_train_loss',
                                     loss.item(), global_step)
                tb_logger.add_scalar('smoothed_train_loss',
                                     train_loss_meter.avg, global_step)

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

                    if global_step % args.save_checkpoints_steps == 0:
                        torch.save(model.state_dict(), os.path.join(
                            args.output_dir, f"checkpoint_{global_step}.pt"))

                    if global_step % args.eval_period == 0:
                        acc = predict(args, model, eval_dataloader,
                                     device, fp16=args.efficient_eval)
                        logger.info("Step %d Train loss %.2f Acc %.2f on epoch=%d" % (
                            global_step, train_loss_meter.avg, acc*100, epoch))

                        tb_logger.add_scalar('dev_acc', acc*100, global_step)

                        # save most recent model
                        torch.save(model.state_dict(), os.path.join(
                            args.output_dir, f"checkpoint_last.pt"))

                        if best_acc < acc:
                            logger.info("Saving model with best  Acc %.2f -> Acc %.2f on epoch=%d" %
                                        (best_acc*100, acc*100, epoch))
                            # model_state_dict = {k: v.cpu() for (
                                # k, v) in model.state_dict().items()}
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_acc = acc
                            wait_step = 0
                            stop_training = False
                        else:
                            wait_step += 1
                            if wait_step == args.wait_step:
                                stop_training = True
                    


            # acc = predict(args, model, eval_dataloader,
            #              device, fp16=args.efficient_eval)
            # tb_logger.add_scalar('dev_acc', acc*100, global_step)
            # logger.info(f"average training loss {train_loss_meter.avg}")
            # if best_acc < acc:
            #     logger.info("Saving model with best  Acc %.2f -> Acc %.2f on epoch=%d" %
            #                 (best_acc*100, acc*100, epoch))
            #     model_state_dict = {k: v.cpu() for (
            #         k, v) in model.state_dict().items()}
            #     torch.save(model_state_dict, os.path.join(
            #         args.output_dir, "best-model.pt"))
            #     model = model.to(device)
            #     best_acc = acc
            #     wait_step = 0

            if stop_training:
                break

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, fp16=args.efficient_eval)
        logger.info(f"test performance {acc}")
        print(acc)


def predict(args, model, eval_dataloader, device, fp16=False):
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
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        if fp16:
            batch_to_feed = convert_to_half(batch_to_feed)
        with torch.no_grad():
            results = model(batch_to_feed)
            product = torch.mm(results["q"], results["c"].t())
            target = torch.arange(product.size(0)).to(product.device)
            prediction = product.argmax(-1)
            pred_res = prediction == target
            num_total += len(pred_res)
            num_correct += sum(pred_res)

    ## linear combination tuning on dev data
    acc = num_correct/num_total
    best_acc = 0
    if acc > best_acc:
        best_acc = acc
    print(f"evaluated {num_total} examples...")
    print(f"avg. Acc: {acc}")


    if fp16:
        model.float()
    model.train()

    return best_acc


if __name__ == "__main__":
    main()
