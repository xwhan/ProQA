from prepro_utils import hash_question, normalize, find_ans_span_with_char_offsets, prepare
import json
from utils import DocDB
from official_eval import normalize_answer
import numpy as np
from tqdm import tqdm
from basic_tokenizer import RegexpTokenizer, SimpleTokenizer

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
import re

import sys
from transformers import BertTokenizer

PROCESS_TOK = None
PROCESS_DB = None
BERT_TOK = None

def init():
    global PROCESS_TOK, PROCESS_DB, BERT_TOK
    PROCESS_TOK = SimpleTokenizer()
    BERT_TOK = BertTokenizer.from_pretrained("bert-base-uncased")
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB('/home/xwhan/code/DrQA/data/wikipedia/nq_paras.db')
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """return all spans that match the pattern"""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        print('Regular expression failed to compile: %s' % pattern)
        return []
    
    matched = [x.group() for x in re.finditer(pattern, text)]
    return list(set(matched))

def para_has_answer(p, answer, tokenizer):
    tokens = tokenizer.tokenize(p)
    text = tokens.words(uncased=True)
    matched = []
    for single_answer in answer:
        single_answer = normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True, tokens.slice(i, i + len(single_answer)).untokenize()
    return False, ""

def match_answer_span(p, answer, tokenizer, match="string"):
    # p has been normalized
    if match == 'string':
        tokens = tokenizer.tokenize(p)
        text = tokens.words(uncased=True)
        matched = set()
        for single_answer in answer:
            single_answer = normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    matched.add(tokens.slice(i, i + len(single_answer)).untokenize())
        return list(matched)
    elif match == 'regex':
        # Answer is a regex
        single_answer = normalize(answer[0])
        return regex_match(p, single_answer)

def process_qa_para(qa_with_result, k=10000, match="string"):
    global PROCESS_DB, PROCESS_TOK
    qa, result = qa_with_result
    matched_paras = {}
    for para_id in result["para_id"][:k]:
        p = PROCESS_DB.get_doc_text(para_id)
        p = normalize(p)
        if match == "string":
            covered, matched = para_has_answer(p, qa["answer"], PROCESS_TOK)
        elif match == "regex":
            single_answer = normalize(qa["answer"][0])
            matched = regex_match(p, single_answer)
            covered = len(matched) > 0
        if covered:
            matched_paras[para_id] = matched
    qa["matched_paras"] = matched_paras
    return qa

def find_span(example):
    global PROCESS_DB, BERT_TOK
    annotated = {}
    for para_id, matched in example["matched_paras"].items():
        p = normalize(PROCESS_DB.get_doc_text(para_id))
        ans_starts, ans_ends, ans_texts = [], [], []
        doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, all_doc_tokens = prepare(
            p, BERT_TOK)
        char_starts = [i for i in range(
            len(p)) if p.startswith(matched, i)]
        assert len(char_starts) > 0
        char_ends = [start + len(matched) - 1 for start in char_starts]
        answer = {"text": matched, "char_spans": list(
            zip(char_starts, char_ends))}
        ans_spans = find_ans_span_with_char_offsets(
            answer, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, BERT_TOK)
        for s, e in ans_spans:
            ans_starts.append(s)
            ans_ends.append(e)
            ans_texts.append(matched)
        annotated[para_id] = {
            # "doc_toks": doc_tokens,
            "doc_subtoks": all_doc_tokens,
            "starts": ans_starts,
            "ends": ans_ends,
            "span_texts": [matched],
            # "tok_to_orig_index": tok_to_orig_index
        }
    example["matched_paras"] = annotated
    return example


def process_ground_paras(retrieved="/data/hongwang/nq_rewrite/db/index_data_new_model/wq_finetuneq_train_10000.txt", save_path="/home/xwhan/retrieval_data/wq_ft_train_matched.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-train.txt", num_workers=40, debug=False, k=10000, match="string"):
    retrieved = [json.loads(l) for l in open(retrieved).readlines()]
    raw_data = [json.loads(l) for l in open(raw_data).readlines()]

    tokenizer = SimpleTokenizer()
    recall = []
    processes = ProcessPool(
        processes=num_workers,
        initializer=init,
    )
    process_qa_para_partial = partial(process_qa_para, k=k, match=match)
    num_tasks = len(raw_data)
    results = []
    for _ in tqdm(processes.imap_unordered(process_qa_para_partial, zip(raw_data, retrieved)), total=len(raw_data)):
        results.append(_)

    topk_covered = [len(r["matched_paras"])>0 for r in results]
    print(np.mean(topk_covered))

    if debug:
        return

    # # annotate those match paras, accelerate training
    # processed = []
    # for _ in tqdm(processes.imap_unordered(find_span, results), total=len(results)):
    #     processed.append(_)

    processes.close()
    processes.join()

    with open(save_path, "w") as g:
        for _ in results:
            g.write(json.dumps(_) + "\n")


def debug(retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_after_ft/wq_finetuneq_dev_5000.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-dev.txt", precomputed="/home/xwhan/retrieval_data/wq_ft_dev_matched.txt", k=10):
    # check wether it reasonable to precompute a paragraph set
    retrieved = [json.loads(l) for l in open(retrieved).readlines()]
    raw_data = [json.loads(l) for l in open(raw_data).readlines()]

    annotated = [json.loads(l) for l in open(precomputed).readlines()]
    qid2goldparas = {hash_question(item["question"]): item["matched_paras"] for item in annotated}

    topk_covered = []
    for qa, result in tqdm(zip(raw_data, retrieved), total=len(raw_data)):
        qid = hash_question(qa["question"])
        covered = 0
        for para_id in result["para_id"][:k]:
            if para_id in qid2goldparas[qid]:
                covered = 1
                break
        topk_covered.append(covered)
    print(np.mean(topk_covered))


if __name__ == "__main__":

    # trec
    process_ground_paras(retrieved="/data/xwhan/data/trec/trec_finetuneq_train-20000.txt", save_path="/home/xwhan/retrieval_data/trec_train_matched_20000.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/trec-train.txt", num_workers=30, k=20000, match="regex")

    # # wq
    # process_ground_paras(retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_after_ft/wq_finetuneq_train-combined_15000.txt",
    #                      save_path="/home/xwhan/retrieval_data/wq_ft_train-combined_matched_15000.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-train-combined.txt", num_workers=30, k=15000)

    # nq
    #process_ground_paras(retrieved="/data/hongwang/nq_rewrite/db/index_data_new_model/nq_finetuneq_train_10000.txt",
    #                     save_path="/home/xwhan/retrieval_data/nq_ft_train_matched.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/nq-train.txt", num_workers=40)


    # # debug
    # process_ground_paras(
    #     retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_new_model/wq_finetuneq_dev_5000_fi.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-dev.txt", debug=True, k=5)
    # process_ground_paras(
    #     retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_new_model/wq_finetuneq_dev.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-dev.txt", debug=True, k=5)
    # process_ground_paras(
    #     retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_new_model/nq_finetuneq_dev_5000_fi.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/nq-dev.txt", debug=True, k=5)
    # process_ground_paras(
    #     retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_new_model/nq_finetuneq_dev.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/nq-dev.txt", debug=True, k=5)
    # #debug(k=30)
    # process_ground_paras(retrieved="/data/hongwang/nq_rewrite/db/index_data_new_model/nq_finetuneq_train_10000.txt",
    #                      save_path="/home/xwhan/retrieval_data/nq_ft_train_matched.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/nq-train.txt", num_workers=40)


    # # debug
    # process_ground_paras(
    #     retrieved="/data/hongwang/nq_rewrite/BERT-QA-Simple/retrieval/index_data_after_ft/wq_finetuneq_dev_5000.txt", raw_data="/home/xwhan/code/DrQA/data/datasets/wq-dev.txt", debug=True, k=30)
    # debug(k=30)

