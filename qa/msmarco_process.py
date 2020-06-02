import json


def extract_qa_p(path="../data/msmarco-qa/train_v2.1.json", output="../data/msmarco-qa/train.txt"):
    data = json.load(open(path))
    data_to_save = []
    for id_, answers in data["answers"].items():
        if answers[0] != 'No Answer Present.':
            passages = data["passages"][id_]
            query = data["query"][id_]
            relevant_p = []
            for p in passages:
                if p["is_selected"]:
                    relevant_p.append(p["passage_text"])
            if len(relevant_p) != 0:
                data_to_save.append({"q": query, "answer": answers, "para": " ".join(relevant_p)})

    with open(output, "w") as g:
        for l in data_to_save:
            g.write(json.dumps(l) + "\n")

from tqdm import tqdm

if __name__ == "__main__":
    # extract_qa_p()

    # data = [json.loads(l)
    #         for l in open("../data/msmarco-qa/dev.txt").readlines()]
    
    # source_file = open("../data/msmarco-qa/val.source", "w")
    # target_file = open("../data/msmarco-qa/val.target", "w")
    # for _ in data:
    #     source_file.write(_["para"] + "\n")
    #     target_file.write(_["q"] + "\n")

    all_paras = [json.loads(l) for l in open(
        "../data/trec-2019/msmarco_paras.txt").readlines()]
    source_file = open("../data/msmarco-qa/test.source", "w")
    for _ in tqdm(all_paras):
        source_file.write(" ".join(_["text"].split()) + "\n")
