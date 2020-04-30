import numpy as np
import json
import faiss
import argparse

"""
retrieval paragraphs given question embed and wikipedia index
"""

def convert_idx2id(idxs):
    idx_id_mapping = json.load(open('index_data/idx_id.json'))
    print(len(idx_id_mapping))
    with open('index_data/retrieve_ids.txt', 'w') as f_out:
        for cand_idx in idxs:
            out_ids = []
            for _ in cand_idx:
                out_ids.append(idx_id_mapping[str(_)])
            f_out.write('<cand_sep>'.join(out_ids)+'\n')

def get_ids(idxs):
    idx_id_mapping = json.load(open('index_data/idx_id.json'))
    ret_ids = []
    for cand_idx in idxs:
        out_ids = []
        for _ in cand_idx:
            out_ids.append(idx_id_mapping[str(_)])
        ret_ids.append(out_ids)
    return ret_ids

def load_question(file_name):
    return [json.loads(_.strip())['question'] for _ in open(file_name).readlines()]

def load_answer(file_name):
    return [json.loads(_.strip())['answer'] for _ in open(file_name).readlines()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('split', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--for-fine-tune', action="store_true", help="generate data for question encoder fine tuning")
    parser.add_argument('--build-qa-data', action="store_true")
    parser.add_argument('--top-5k-path', type=str,
                        default="/data/hongwang/nq_rewrite/db/index_data_new_model/nq_finetuneq_train_with_labels.txt")
    args = parser.parse_args()

    d = 128
    # xq = np.load("index_data/nq_dev_query_embed.npy").astype('float32')
    xq = np.load(
        f'/data/xwhan/data/{args.dataset}/{args.dataset}_{args.split}_query_embed.npy').astype('float32')
    xb = np.load(args.indexpath).astype('float32')

    index = faiss.IndexFlatIP(d)   # build the index
    index.add(xb)                  # add vectors to the index
    D, I = index.search(xq, args.topk)     # actual search

    if args.for_fine_tune:
        ids = get_ids(I)
        questions = load_question(
            f'/home/xwhan/code/DrQA/data/datasets/{args.dataset}-{args.split}.txt')
        with open(f'/data/xwhan/data/{args.dataset}/{args.dataset}_finetuneq_{args.split}-{args.topk}.txt', 'w') as f_out:
            for i, question in enumerate(questions):
                f_out.write(json.dumps({'question': questions[i], 'para_id': ids[i], 'para_embed_idx': I[i].tolist()})+'\n')
    elif args.build_qa_data:
        ids = get_ids(I)

        if args.top_5k_path != "":
            data = [json.loads(l) for l in open(args.top_5k_path).readlines()]
    
        questions = load_question(
            f'/home/xwhan/code/DrQA/data/datasets/{args.dataset}-{args.split}.txt')
        answers = load_answer(
            f'/home/xwhan/code/DrQA/data/datasets/{args.dataset}-{args.split}.txt')
        with open(f"/data/xwhan/data/{args.dataset}/{args.dataset}-{args.split}-dense-retrieve-qa-top{args.topk}.txt", "w") as g:
            for i, q in enumerate(questions):
                if args.top_5k_path != "":
                    assert q == data[i]["question"]
                    g.write(json.dumps(
                        {'question': questions[i], 'answer': answers[i], 'para_id': ids[i], 'para_embed_idx': data[i]["para_embed_idx"], 'para_labels': data[i]["para_labels"]})+'\n')
                else:
                    g.write(json.dumps({'question': questions[i], 'answer': answers[i], 'para_id': ids[i], 'para_embed_idx': I[i].tolist()})+'\n')
    else:
        convert_idx2id(I)

