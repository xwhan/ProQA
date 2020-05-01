import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import QADataset, collate
from bert_qa import BertForQuestionAnswering
from transformers import AdamW, BertConfig, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from eval_utils import get_final_text
from official_eval import metric_max_over_ground_truths, f1_score, exact_match_score

from utils import move_to_cuda, convert_to_half, AverageMeter
from config import get_args

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
    model_name = f"{data_name}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-{args.prefix}-lr{args.learning_rate}-{args.bert_model_name}"
    tb_logger = SummaryWriter(os.path.join(args.output_dir, "tflogs", model_name))
    args.output_dir = os.path.join(args.output_dir, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
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
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

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
    model = BertForQuestionAnswering(bert_config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    if args.do_train and args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    eval_dataset = QADataset(
        tokenizer, args.predict_file, args.max_query_length, args.max_seq_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate, pin_memory=True)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint is not None:
        logger.info("Loading from {}".format(args.init_checkpoint))
        if args.do_train and args.init_checkpoint == "":
            model = BertForQuestionAnswering.from_pretrained(
                args.bert_model_name)
        else:
            state_dict = torch.load(args.init_checkpoint)
            filter = lambda x: x[7:] if x.startswith('module.') else x
            state_dict = {filter(k):v for (k,v) in state_dict.items()}
            model.load_state_dict(state_dict)
    model.to(device)

    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

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
        global_step = 0
        best_f1 = (-1, -1)
        wait_step = 0
        stop_training = False
        train_loss_meter = AverageMeter()
        logger.info('Start training....')
        model.train()
        train_dataset = QADataset(tokenizer, args.train_file, args.max_query_length, args.max_seq_length)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, collate_fn=collate, shuffle=True, pin_memory=True)

        for epoch in range(int(args.num_train_epochs)):

            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = move_to_cuda(batch)
                outputs = model(batch["net_input"])
                loss = outputs["span_loss"]

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                train_loss_meter.update(loss.item())
                tb_logger.add_scalar('batch_train_loss', loss.item(), global_step)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

                    if global_step % args.eval_period == 0:
                        f1 = predict(logger, args, model, eval_dataloader, device, fp16=args.efficient_eval)
                        logger.info("Step %d Train loss %.2f EM %.2f F1 %.2f on epoch=%d" % (
                            global_step, train_loss_meter.avg, f1[0]*100, f1[1]*100, epoch))

                        tb_logger.add_scalar('dev_f1', f1[0]*100, global_step)
                        tb_logger.add_scalar('dev_em', f1[1]*100, global_step)

                        if best_f1 < f1:
                            logger.info("Saving model with best EM: %.2f (F1 %.2f) -> %.2f (F1 %.2f) on epoch=%d" % \
                                    (best_f1[1]*100, best_f1[0]*100, f1[1]*100, f1[0]*100, epoch))
                            model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                            torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                            model = model.to(device)
                            best_f1 = f1
                            wait_step = 0
                            stop_training = False
                        else:
                            wait_step += 1
                            if wait_step == args.wait_step:
                                stop_training = True

            f1 = predict(logger, args, model, eval_dataloader,
                            device, fp16=args.efficient_eval)
            logger.info("Step %d Train loss %.2f EM %.2f F1 %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, f1[0]*100, f1[1]*100, epoch))
            tb_logger.add_scalar('dev_f1', f1[0]*100, global_step)
            tb_logger.add_scalar('dev_em', f1[1]*100, global_step)
            logger.info(f"average training loss {train_loss_meter.avg}")


            if stop_training:
                break

        logger.info("Training finished!")

    # elif args.do_predict:
    #     if type(model)==list:
    #         model = [m.eval() for m in model]
    #     else:
    #         model.eval()
    #     f1 = predict(logger, args, model, eval_dataloader, eval_examples, eval_features,
    #                  device, fp16=args.efficient_eval, write_prediction=False)
    #     logger.info(f"test performance {f1}")
    #     print(f1)


def predict(logger, args, model, eval_dataloader, device, fp16=False):
    model.eval()
    all_results = []

    if fp16:
        model.half()

    qid2results = {}
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch["net_input"])
        if fp16:
            batch_to_feed = convert_to_half(batch_to_feed)
        with torch.no_grad():
            results = model(batch_to_feed)
            batch_start_logits = results["start_logits"]
            batch_end_logits = results["end_logits"]
            question_mask = batch_to_feed["paragraph_mask"].ne(1)
            outs = [o.float().masked_fill(question_mask, -1e10).type_as(o)
                    for o in [batch_start_logits, batch_end_logits]]

        span_scores = outs[0][:,:,None] + outs[1][:,None]
        max_answer_lens = 20
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), max_answer_lens)
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 - 
            span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)

        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]
        end_position = span_scores_masked.max(dim=2)[1].gather(1, start_position.unsqueeze(1)).squeeze(1)

        para_offset = batch['para_offset']
        start_position_ = list(np.array(start_position.tolist()) - np.array(para_offset))
        end_position_ = list(np.array(end_position.tolist()) - np.array(para_offset))

        for idx, qid in enumerate(batch['id']):
            start = start_position_[idx]
            end = end_position_[idx]
            tok_to_orig_index = batch['tok_to_orig_index'][idx]
            doc_tokens = batch['doc_tokens'][idx]
            wp_tokens = batch['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text, logger, do_lower_case=args.do_lower_case, verbose_logging=False)
            qid2results[qid] = [final_text, batch['true_answers'][idx]]

    f1s = [metric_max_over_ground_truths(f1_score, item[0], item[1]) for item in qid2results.values()]
    ems = [metric_max_over_ground_truths(exact_match_score, item[0], item[1]) for item in qid2results.values()]

    print(f"evaluated {len(f1s)} examples...")
    if fp16:
        model.float()
    model.train()

    return (np.mean(f1s), np.mean(ems))


if __name__ == "__main__":
    main()
