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
from torch.utils.data.distributed import DistributedSampler
from bert_retrieve_qa import BertRetrieveQA
from transformers import AdamW, BertConfig, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from eval_utils import get_final_text
from official_eval import metric_max_over_ground_truths, exact_match_score, regex_match_score
from online_sampler import OnlineSampler


from utils import move_to_cuda, convert_to_half, AverageMeter, DocDB
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

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # tb logger
    data_name = args.train_file.split("/")[-1].split('-')[0]
    model_name = f"dense-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-{args.prefix}-lr{args.learning_rate}-{args.bert_model_name}-qdrop{args.qa_drop}-sn{args.shared_norm}-sep{args.separate}-as{args.add_select}-noearly{args.drop_early}"
    if args.do_train:
        tb_logger = SummaryWriter(os.path.join(
            args.output_dir, "tflogs", "dense", model_name))
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
    model = BertRetrieveQA(bert_config, args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    logger.info("Loading para db and pretrained index ...")
    para_db = DocDB(args.db_path)
    para_embed = np.load(args.index_path).astype('float32')

    if args.do_train and args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))
    
    exact_search = True if args.do_predict else False
    eval_dataloader = OnlineSampler(args.raw_eval_data, tokenizer, args.max_query_length,
                                    args.max_seq_length, para_db, para_embed, exact_search=exact_search, cased=args.use_spanbert, regex=args.regex)

    if args.init_checkpoint != "":
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    logger.info(
        f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.fix_para_encoder:
        model.freeze_c_encoder()
    
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
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_em = 0
        wait_step = 0
        stop_training = False
        train_loss_meter = AverageMeter()
        logger.info('Start training....')
        model.train()
        train_dataloader = OnlineSampler(
            args.raw_train_data, tokenizer, args.max_query_length, args.max_seq_length, para_db, para_embed, matched_para_path=args.matched_para_path, cased=args.use_spanbert, regex=args.regex)
        for epoch in range(int(args.num_train_epochs)):       
            train_dataloader.shuffle()
            failed_retrival = 0
            for batch in tqdm(train_dataloader.load(model.retriever, k=args.train_batch_size), total=len(train_dataloader)):
                batch_step += 1
                if batch == {}:
                    failed_retrival += 1
                    continue
                batch = move_to_cuda(batch)
                outputs = model(batch["net_input"])
                loss = outputs["loss"]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
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

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        em = predict(args, model, eval_dataloader,
                                     device, fp16=args.efficient_eval)
                        logger.info("Step %d Train loss %.2f EM %.2f on epoch=%d" % (
                            global_step, train_loss_meter.avg, em*100, epoch))

                        tb_logger.add_scalar('dev_em', em*100, global_step)

                        if best_em < em:
                            logger.info("Saving model with best EM: %.2f -> EM %.2f on epoch=%d" %
                                        (best_em*100, em*100, epoch))
                            model_state_dict = {k: v.cpu() for (
                                k, v) in model.state_dict().items()}
                            torch.save(model_state_dict, os.path.join(
                                args.output_dir, "best-model.pt"))
                            model = model.to(device)
                            best_em = em
                            wait_step = 0
                            stop_training = False
                        else:
                            wait_step += 1
                            if wait_step == args.wait_step:
                                stop_training = True

            logger.info(f"Failed retrieval: {failed_retrival}/{len(train_dataloader)} ...")
            em = predict(args, model, eval_dataloader,
                         device, fp16=args.efficient_eval)
            tb_logger.add_scalar('dev_em', em*100, global_step)
            logger.info(f"average training loss {train_loss_meter.avg}")
            if best_em < em:
                logger.info("Saving model with best EM: %.2f  -> %.2f on epoch=%d" %
                            (best_em*100, em*100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, "best-model.pt"))
                model = model.to(device)
                best_em = em
                wait_step = 0

            if epoch > 15:
                logger.info(f"Saving model after epoch {epoch + 1}")
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"model-{epoch+1}-{em}.pt"))

            if stop_training:
                break

        logger.info("Training finished!")

    elif args.do_predict:
        f1 = predict(args, model, eval_dataloader,
                     device, fp16=args.efficient_eval)
        logger.info(f"test performance {f1}")
        print(f1)


def predict(args, model, eval_dataloader, device, fp16=False):
    model.eval()
    if fp16:
        model.half()

    all_results = []
    PredictionMeta = collections.namedtuple(
        "Prediction", ["text", "rank_score", "passage", "span_score", "question"])
    qid2results = defaultdict(list)
    qid2ground = {}

    for batch in tqdm(eval_dataloader.eval_load(model.retriever, args.eval_k), total=len(eval_dataloader)):
        
        batch_to_feed = move_to_cuda(batch["net_input"])
        if fp16:
            batch_to_feed = convert_to_half(batch_to_feed)
        with torch.no_grad():
            results = model(batch_to_feed)
            batch_start_logits = results["start_logits"]
            batch_end_logits = results["end_logits"]
            batch_rank_logits = results["rank_logits"]
            if args.add_select:
                batch_select_logits = results["select_logits"]

            outs = [batch_start_logits, batch_end_logits]

        span_scores = outs[0][:, :, None] + outs[1][:, None]
        max_answer_lens = 10
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(
            np.triu(np.ones((max_seq_len, max_seq_len)), 0), max_answer_lens)
        span_mask = span_scores.data.new(
            max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 -
                                                              span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)

        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]
        end_position = span_scores_masked.max(dim=2)[1].gather(
            1, start_position.unsqueeze(1)).squeeze(1)

        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist()

        if args.add_select:
            rank_logits = batch_select_logits.view(-1).tolist()
        else:
            rank_logits = batch_rank_logits.view(-1).tolist()

        para_offset = batch['para_offset']
        start_position_ = list(
            np.array(start_position.tolist()) - np.array(para_offset))
        end_position_ = list(
            np.array(end_position.tolist()) - np.array(para_offset))

        for idx, qid in enumerate(batch['id']):
            start = start_position_[idx]
            end = end_position_[idx]
            rank_score = rank_logits[idx]
            span_score = answer_scores[idx]
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
            final_text = get_final_text(
                tok_text, orig_text, do_lower_case=args.do_lower_case, verbose_logging=False)
            question = batch["q"][idx]
            qid2results[qid].append(
                PredictionMeta(
                    text=final_text,
                    rank_score=rank_score,
                    span_score=span_score,
                    passage=" ".join(doc_tokens),
                    question=question,
                )
            )
            qid2ground[qid] = batch["true_answers"][idx]

    if args.save_all:
        print("Saving all prediction results ...")
        with open(f"{args.prefix}_all.json", "w") as g:
            json.dump(qid2results, g)
        with open(f"{args.prefix}_ground.json", "w") as g:
            json.dump(qid2ground, g)

    ## linear combination tuning on dev data
    best_em = 0
    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1]:
        results_to_save = []
        ems = []
        for qid in qid2results.keys():
            qid2results[qid] = sorted(
                qid2results[qid], key=lambda x: alpha*x.span_score + (1 - alpha)*x.rank_score, reverse=True)
            match_fn = regex_match_score if args.regex else exact_match_score
            ems.append(metric_max_over_ground_truths(
                match_fn, qid2results[qid][0].text, qid2ground[qid]))
            results_to_save.append({
                "question": qid2results[qid][0].question,
                "para": qid2results[qid][0].passage,
                "answer": qid2results[qid][0].text,
                "rank_score": qid2results[qid][0].rank_score,
                "gold": qid2ground[qid],
                "em": ems[-1]
            })
        em = np.mean(ems)
        if em > best_em:
            best_em = em
        print(f"evaluated {len(ems)} examples...")
        print(f"alpha: {alpha}; avg. EM: {em}")

        if args.save_pred:
            with open(f"{args.prefix}_{alpha}.json", "w") as g:
                for line in results_to_save:
                    g.write(json.dumps(line) + "\n")

    if type(model) != list:
        if fp16:
            model.float()
        model.train()

    return best_em


if __name__ == "__main__":
    main()

