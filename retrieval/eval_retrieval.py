
import numpy as np
import json
import faiss
import argparse

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import defaultdict

from basic_tokenizer import SimpleTokenizer
from utils import DocDB, normalize


PROCESS_TOK = None
PROCESS_DB = None

def init(db_path):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = SimpleTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = DocDB(db_path)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def para_has_answer(answer, para, return_matched=False):
    global PROCESS_DB, PROCESS_TOK
    text = normalize(para)
    tokens = PROCESS_TOK.tokenize(text)
    text = tokens.words(uncased=True)
    assert len(text) == len(tokens)
    for single_answer in answer:
        single_answer = normalize(single_answer)
        single_answer = PROCESS_TOK.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                if return_matched:
                    return True, tokens.slice(i, i + len(single_answer)).untokenize()
                else:
                    return True
    if return_matched:
        return False, ""
    return False

def get_score(answer_doc, topk=80):
    """Search through all the top docs to see if they have the answer."""
    question, answer, doc_ids = answer_doc
    top5doc_covered = 0
    global PROCESS_DB
    all_paras = [PROCESS_DB.get_doc_text(doc_id) for doc_id in doc_ids]

    topk_paras = all_paras[:topk]
    topkpara_covered = []
    for p in topk_paras:
        topkpara_covered.append(int(para_has_answer(answer, p)))

    return {
        str(topk): int(np.sum(topkpara_covered) > 0),
        "5": int(np.sum(topkpara_covered[:5]) > 0),
        "10": int(np.sum(topkpara_covered[:10]) > 0),
        "20": int(np.sum(topkpara_covered[:20]) > 0),
        "50": int(np.sum(topkpara_covered[:50]) > 0),
    }


def convert_idx2id(idxs):
    idx_id_mapping = json.load(open('../pretrained_models/idx_id.json'))
    retrieval_results = []
    for cand_idx in idxs:
        out_ids = []
        for _ in cand_idx:
            out_ids.append(idx_id_mapping[str(_)])
        retrieval_results.append(out_ids)
    return retrieval_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('query_embed', type=str, default=None)
    parser.add_argument('db', type=str, default=None)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--num-workers', type=int, default=10)
    args = parser.parse_args()

    qas = [json.loads(line) for line in open(args.raw_data).readlines()]
    questions = [item["question"] for item in qas]
    answers = [item["answer"] for item in qas]

    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=[args.db]
    )

    d = 128
    xq = np.load(args.query_embed).astype('float32')
    xb = np.load(args.indexpath).astype('float32')

    index = faiss.IndexFlatIP(d)   # build the index
    index.add(xb)                  # add vectors to the index
    D, I = index.search(xq, args.topk)     # actual search

    retrieval_results = convert_idx2id(I)

    assert len(retrieval_results) == len(questions) == len(answers)
    answers_docs = zip(questions, answers, retrieval_results)

    get_score_partial = partial(
         get_score, topk=args.topk)
    results = processes.map(get_score_partial, answers_docs)

    aggregate = defaultdict(list)
    for r in results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        results = aggregate[k]
        print('Top {} Recall for {} QA pairs: {} ...'.format(
            k, len(results), np.mean(results)))
