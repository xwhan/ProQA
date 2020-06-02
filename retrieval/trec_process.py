import json
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import faiss

def prepare_corpus(path="../data/trec-2019/collection.tsv", save_path="../data/trec-2019/msmarco_paras.txt"):
    corpus = []
    for line in tqdm(open(path).readlines()):
        line = line.strip()
        pid, text = line.split("\t")
        corpus.append({"text": text, "id": int(pid)})
    with open(save_path, "w") as g:
        for _ in corpus:
            g.write(json.dumps(_) + "\n")

def extract_labels(
    input="../data/trec-2019/qrels.train.tsv", 
    output="../data/trec-2019/msmacro-train.txt",
    queries="../data/trec-2019/queries.train.tsv"
    ):
    # id2queries 
    qid2query = {}
    for line in open(queries).readlines():
        line = line.strip()
        qid, q = line.split("\t")[0], line.split("\t")[1]
        if q.endswith("?"):
            q = q[:-1]
        qid2query[int(qid)] = q
    print(len(qid2query))
    
    # queries with groundtruths
    qid2ground = defaultdict(list)
    for line in open(input).readlines():
        line = line.strip()
        qid, pid = line.split("\t")[0], line.split("\t")[2]
        qid2ground[int(qid)].append(int(pid))
    print(len(qid2ground))

    # generate data for train/dev
    with open(output, "w") as g:
        for qid, labels in qid2ground.items():
            question = qid2query[qid]
            sample = {"question":question, "labels": labels, "qid": qid}
            g.write(json.dumps(sample) + "\n")


def debug():
    top1000_dev = open("../data/trec-2019/top1000.dev").readlines()
    qid2top10000 = defaultdict(list)
    for l in top1000_dev:
        qid2top10000[int(l.split("\t")[0])].append(int(l.split("\t")[1]))
    print(len(qid2top10000))

    processed_dev = [json.loads(l) for l in tqdm(open(
        "../data/trec-2019/processed/dev.txt").readlines())]
    qid2ground = {_["qid"]: _["labels"] for _ in processed_dev}

    covered = []
    for qid in qid2top10000.keys():
        top1000_labels = [int(_ in qid2ground[qid]) for _ in qid2top10000[qid]]
        covered.append(int(np.sum(top1000_labels) > 0))

    print(len(covered))
    print(np.mean(covered))


def retrieve_topk(index_path="../data/trec-2019/embeds/msmarco_paras_embed.npy", query_embeds="../data/trec-2019/embeds/msmarco-train-query.npy", query_input="../data/trec-2019/msmacro-train.txt", output="../data/trec-2019/processed/train.txt"):
    d = 128
    xq = np.load(query_embeds).astype('float32')
    xb = np.load(index_path).astype('float32')

    index = faiss.IndexFlatIP(d)   # build the index
    index.add(xb)                  # add vectors to the index
    D, I = index.search(xq, 10000)     # actual search

    raw_data = [json.loads(l) for l in open(query_input).readlines()]

    processed = []
    covered = []
    for idx, para_indice in enumerate(I):
        orig_sample = raw_data[idx]
        para_embed_idx = [int(_) for _ in para_indice]
        para_labels = [int(_ in orig_sample["labels"]) for _ in para_embed_idx]
        orig_sample["para_embed_idx"] = para_embed_idx
        orig_sample["para_labels"] = para_labels
        processed.append(orig_sample)
        covered.append(int(np.sum(para_labels) > 0))
    
    print(f"Avg recall: {np.mean(covered)}")
    with open(output, "w") as g:
        for _ in processed:
            g.write(json.dumps(_) + "\n")


if __name__ == "__main__":
    # prepare_corpus()
    # extract_labels(input="../data/trec-2019/qrels.dev.small.tsv",
    #                output="../data/trec-2019/msmacro-dev-small.txt",
    #                queries="../data/trec-2019/queries.dev.tsv")

    # debug()

    retrieve_topk()
