import json
from tokenizer import _is_whitespace, _is_punctuation, process, whitespace_tokenize
from transformers import BertTokenizer
from tqdm import tqdm
from multiprocessing import Pool
import hashlib
import unicodedata
import re
import sys
import numpy as np

def hash_question(q):
    hash_object = hashlib.md5(q.encode())
    return hash_object.hexdigest()

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def load_mrqa_dataset(path):
    raw_data = [json.loads(line.strip()) for line in open(path).readlines()[1:]]

    qa_data = []
    for item in raw_data:
        id_ = item["id"]
        context = item["context"]
        for qa in item["qas"]:
            qid = qa["qid"]
            question = qa["question"]
            answers = qa.get("answers", [])
            matched_answers = qa.get("detected_answers", [])
            qa_data.append(
                {
                    "qid": qid,
                    "question": question,
                    "context": context,
                    "matched_answers": matched_answers,
                    "true_answers": answers
                }
            )
    return qa_data


def load_openqa_dataset(path, filter_no_answer=False):

    def _check_no_ans(sample):
        no_ans = True
        for para in sample["retrieved"]:
            if para["matched_answer"] != "":
                no_ans = False
                return no_ans
        return no_ans

    raw_data = [json.loads(line.strip()) for line in open(path).readlines()]

    if filter_no_answer:
        raw_data = [item for item in raw_data if not _check_no_ans(item)]

    print(f"Loading {len(raw_data)} QA pairs")
    return raw_data

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                            orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def find_ans_span_with_char_offsets(detected_ans, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer):
    # could return mutiple spans for an answer string
    ans_text = detected_ans["text"]
    char_spans = detected_ans["char_spans"]
    ans_subtok_spans = []
    for char_start, char_end in char_spans:
        tok_start = char_to_word_offset[char_start]
        tok_end = char_to_word_offset[char_end] #  char_end points to the last char of the answer, not one after
        sub_tok_start = orig_to_tok_index[tok_start]

        if tok_end < len(doc_tokens) - 1:
            sub_tok_end = orig_to_tok_index[tok_end + 1] - 1
        else:
            sub_tok_end = len(all_doc_tokens) - 1

        actual_text = " ".join(doc_tokens[tok_start:(tok_end + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(ans_text))
        if actual_text.find(cleaned_answer_text) == -1:
            print("Could not find answer: '{}' vs. '{}'".format(
                actual_text, cleaned_answer_text))

        (sub_tok_start, sub_tok_end) = _improve_answer_span(
            all_doc_tokens, sub_tok_start, sub_tok_end, tokenizer, ans_text)
        ans_subtok_spans.append((sub_tok_start, sub_tok_end))

    return ans_subtok_spans

def tokenize_item(sample, tokenizer):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in sample["context"]:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    orig_to_tok_index = []
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = process(token, tokenizer)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    q_sub_toks = process(sample["question"], tokenizer)

    # finding answer spans
    ans_starts, ans_ends, ans_texts = [], [], []
    for answer in sample["matched_answers"]:
        ans_spans = find_ans_span_with_char_offsets(
            answer, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer)

        for (s, e) in ans_spans:
            ans_starts.append(s)
            ans_ends.append(e)
            ans_texts.append(answer["text"])

    return {
        "q_subtoks": q_sub_toks,
        "qid": sample["qid"],
        "doc_toks": doc_tokens,
        "doc_subtoks": all_doc_tokens,
        "tok_to_orig_index": tok_to_orig_index,
        "starts": ans_starts,
        "ends": ans_ends,
        "span_texts": ans_texts,
        "true_answers": sample["true_answers"]
    }

def prepare(context, tokenizer):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    for c in context:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    orig_to_tok_index = []
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, all_doc_tokens

def tokenize_item_openqa(sample, tokenizer):
    """
    process all the retrieved paragraphs of a QA pair
    """ 
    q_sub_toks = process(sample["question"], tokenizer)

    examples = []
    for para_idx, para in enumerate(sample["retrieved"]):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        context = normalize(para["para"])

        for c in context:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        orig_to_tok_index = []
        tok_to_orig_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = process(token, tokenizer)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # finding answer spans
        ans_starts, ans_ends, ans_texts = [], [], []
        no_answer = 0
        if para["matched_answer"] == "":
            ans_starts.append(-1)
            ans_ends.append(-1)
            ans_texts.append("")
            no_answer = 1
        else:
            ans_texts.append(para["matched_answer"])
            char_starts = [i for i in range(
                len(context)) if context.startswith(para["matched_answer"], i)]

            if len(char_starts) == 0:
                import pdb; pdb.set_trace()
            char_ends = [start + len(para["matched_answer"]) - 1 for start in char_starts]
            answer = {"text": para["matched_answer"], "char_spans": list(zip(char_starts, char_ends))}
            ans_spans = find_ans_span_with_char_offsets(
                answer, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer)
            for (s, e) in ans_spans:
                ans_starts.append(s)
                ans_ends.append(e)
                ans_texts.append(answer["text"])
        qid = hash_question(sample["question"])

        examples.append({
            "q": sample["question"],
            "q_subtoks": q_sub_toks,
            "qid": qid,
            "para_id": para_idx,
            "doc_toks": doc_tokens,
            "doc_subtoks": all_doc_tokens,
            "tok_to_orig_index": tok_to_orig_index,
            "starts": ans_starts,
            "ends": ans_ends,
            "span_texts": ans_texts,
            "true_answers": sample["gold_answer"],
            "no_answer": no_answer,
            "bm25": para["bm25"],
        })
    
    return examples

def tokenize_items(items, tokenizer, verbose=False, openqa=False):
    if verbose:
        items = tqdm(items)
    if openqa:
        results = []
        for _ in items:
            results.extend(tokenize_item_openqa(_, tokenizer))
        return results
    else:
        return [tokenize_item(_, tokenizer) for _ in items]

def tokenize_data(dataset, bert_model_name="bert-large-cased-whole-word-masking", num_workers=10, save_path=None, openqa=False):

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    chunk_size = len(dataset) // num_workers
    offsets = [
        _ * chunk_size for _ in range(0, num_workers)] + [len(dataset)]
    pool = Pool(processes=num_workers)
    print(f'Start multi-processing with {num_workers} workers....')
    results = [pool.apply_async(tokenize_items, args=(
        dataset[offsets[work_id]: offsets[work_id + 1]], tokenizer, True, openqa)) for work_id in range(num_workers)]
    outputs = [p.get() for p in results]
    samples = []
    for o in outputs:
        samples.extend(o)

    # check the average number of matched spans
    answer_nums = [len(item["starts"])
                   for item in samples if item["no_answer"] == 0]
    print(f"Average number of matched answers: {np.mean(answer_nums)}...")
    print(f"Processed {len(samples)} examples...")
    if save_path:
        with open(save_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    else:
        return samples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="bert-base-uncased", type=str)
    parser.add_argument("--data", default="wq", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--filter", action="store_true", help="whether to filter no-answer QA pair")
    parser.add_argument("--dense-index", action="store_true")
    args = parser.parse_args()

    filter_ = True if "train" in args.split else False

    if args.dense_index:
        train_raw = load_openqa_dataset(
            f"../data/{args.data}/{args.data}-{args.split}-dense-final.txt", filter_no_answer=filter_)
        save_path = f"../data/{args.data}/{args.data}-{args.split}-dense-filtered-tokenized.txt" if filter_ else \
            f"../data/{args.data}/{args.data}-{args.split}-dense-tokenized.txt"
    else:
        train_raw = load_openqa_dataset(
            f"../data/{args.data}/{args.data}-{args.split}-openqa-p{args.topk}.txt", filter_no_answer=filter_)
        save_path = f"../data/{args.data}/{args.data}-{args.split}-openqa-filtered-tokenized-p{args.topk}-all-matched.txt" if filter_ else \
            f"../data/{args.data}/{args.data}-{args.split}-openqa-tokenized-p{args.topk}-all-matched.txt"

    train_tokenized = tokenize_data(train_raw, bert_model_name=args.model_name, save_path=save_path, openqa=True, num_workers=10)
