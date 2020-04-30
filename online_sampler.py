import torch
import json
import numpy as np
import random
from prepro_utils import hash_question, normalize, find_ans_span_with_char_offsets, prepare
from utils import DocDB
import faiss
from official_eval import normalize_answer
from basic_tokenizer import SimpleTokenizer
from prepro_dense import para_has_answer, match_answer_span
from tqdm import tqdm

from transformers import BertTokenizer

"""
retrieve paragraphs and find span for top5 on the fly
"""


def normalize_para(s):

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class OnlineSampler(object):
    
    def __init__(self, 
        raw_data, 
        tokenizer,
        max_query_length,
        max_length,
        db,
        para_embed,
        index2paraid='retrieval/index_data/idx_id.json',
        matched_para_path="",
        exact_search=False,
        cased=False,
        regex=False
        ):

        self.max_length = max_length
        self.max_query_length = max_query_length
        self.para_embed = para_embed
        self.cased = cased # spanbert used cased tokenization
        self.regex = regex

        if self.cased:
            self.cased_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # if not exact_search:
        quantizer = faiss.IndexFlatIP(128)
        self.index = faiss.IndexIVFFlat(quantizer, 128, 100)
        self.index.train(self.para_embed)
        self.index.add(self.para_embed)
        self.index.nprobe = 20
        # else:
        #     self.index = faiss.IndexFlatIP(128)
        #     self.index.add(self.para_embed)

        self.tokenizer = tokenizer
        self.qa_data = [json.loads(l) for l in open(raw_data).readlines()]
        self.index2paraid = json.load(open(index2paraid))
        self.para_db = db
        self.matched_para_path = matched_para_path
        if self.matched_para_path != "":
            print(f"Load matched gold paras from {self.matched_para_path}")
            annotated = [json.loads(l) for l in tqdm(open(
                self.matched_para_path).readlines())]
            self.qid2goldparas = {hash_question(
                item["question"]): item["matched_paras"] for item in annotated}

        self.basic_tokenizer = SimpleTokenizer()

    def shuffle(self):
        random.shuffle(self.qa_data)

    def __len__(self):
        return len(self.qa_data)
    
    def load(self, retriever, k=5):
        for qa in self.qa_data:
            with torch.no_grad():
                q_ids = torch.LongTensor(self.tokenizer.encode(
                    qa["question"], max_length=self.max_query_length)).view(1,-1).cuda()
                q_masks = torch.ones(q_ids.shape).bool().view(1,-1).cuda()
                q_cls = retriever.bert_q(q_ids, q_masks)[1]
                q_embed = retriever.proj_q(q_cls).data.cpu().numpy().astype('float32')

            _, I = self.index.search(q_embed, 5000) # retrieve
            para_embed_idx = I.reshape(-1)

            if self.cased:
                q_ids_cased = torch.LongTensor(self.cased_tokenizer.encode(
                    qa["question"], max_length=self.max_query_length)).view(1, -1)

            para_idx = [self.index2paraid[str(_)] for _ in para_embed_idx]
            para_embeds = self.para_embed[para_embed_idx]

            qid = hash_question(qa["question"])
            gold_paras = self.qid2goldparas[qid]

            # match answer strings
            p_labels = []
            batched_examples = []
            topk5000_labels = [int(_ in gold_paras) for _ in para_idx]

            # match answer spans in top5 paras
            for p_idx in para_idx[:k]:
                p = normalize(self.para_db.get_doc_text(p_idx))
                # p_covered, matched_string = para_has_answer(p, qa["answer"], self.basic_tokenizer)
                matched_spans = match_answer_span(
                    p, qa["answer"], self.basic_tokenizer, match="regex" if self.regex else "string")
                p_covered = int(len(matched_spans) > 0)
                ans_starts, ans_ends, ans_texts = [], [], []

                if self.cased:
                    doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, all_doc_tokens = prepare(p, self.cased_tokenizer)
                else:
                    doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, all_doc_tokens = prepare(
                        p, self.tokenizer)

                if p_covered:
                    for matched_string in matched_spans:
                        char_starts = [i for i in range(
                            len(p)) if p.startswith(matched_string, i)]
                        if len(char_starts) > 0:
                            char_ends = [start + len(matched_string) - 1 for start in char_starts]
                            answer = {"text": matched_string, "char_spans": list(
                                zip(char_starts, char_ends))}

                            if self.cased:
                                ans_spans = find_ans_span_with_char_offsets(answer, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, self.cased_tokenizer)
                            else:
                                ans_spans = find_ans_span_with_char_offsets(
                                    answer, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, self.tokenizer)

                            for s, e in ans_spans:
                                ans_starts.append(s)
                                ans_ends.append(e)
                                ans_texts.append(matched_string)
                batched_examples.append({
                    "qid": hash_question(qa["question"]),
                    "q": qa["question"],
                    "true_answers": qa["answer"],
                    "doc_subtoks": all_doc_tokens,
                    "starts": ans_starts,
                    "ends": ans_ends,
                    "covered": p_covered 
                })

                # # look up saved
                # if p_idx in gold_paras:
                #     p_covered = 1
                #     all_doc_tokens = gold_paras[p_idx]["doc_subtoks"]
                #     ans_starts = gold_paras[p_idx]["starts"]
                #     ans_ends = gold_paras[p_idx]["ends"]
                #     ans_texts = gold_paras[p_idx]["span_texts"]
                # else:
                #     p_covered = 0
                #     p = normalize(self.para_db.get_doc_text(p_idx))
                #     _, _, _, _, all_doc_tokens = prepare(p, self.tokenizer)
                #     ans_starts, ans_ends, ans_texts = [], [], []

                # batched_examples.append({
                #     "qid": hash_question(qa["question"]),
                #     "q": qa["question"],
                #     "true_answers": qa["answer"],
                #     "doc_subtoks": all_doc_tokens,
                #     "starts": ans_starts,
                #     "ends": ans_ends,
                #     "covered": p_covered 
                # })
                p_labels.append(int(p_covered))

            # calculate loss only when the top5000 covered the answer passage
            if np.sum(topk5000_labels) > 0 or np.sum(p_labels) > 0:
                # training tensors
                for item in batched_examples:
                    item["input_ids_q"] = q_ids.view(-1).cpu()

                    if self.cased:
                        item["input_ids_q_cased"] = q_ids_cased.view(-1)
                        para_offset = item["input_ids_q_cased"].size(0)
                    else:
                        para_offset = item["input_ids_q"].size(0)

                    max_toks_for_doc = self.max_length - para_offset - 1
                    para_subtoks = item["doc_subtoks"]
                    if len(para_subtoks) > max_toks_for_doc:
                        para_subtoks = para_subtoks[:max_toks_for_doc]
                    
                    if self.cased:
                        p_ids = self.cased_tokenizer.convert_tokens_to_ids(para_subtoks)
                    else:
                        p_ids = self.tokenizer.convert_tokens_to_ids(
                            para_subtoks)
                    item["input_ids_c"] = self._add_special_token(torch.LongTensor(p_ids))
                    paragraph = item["input_ids_c"][1:-1]
                    if self.cased:
                        item["input_ids"], item["segment_ids"] = self._join_sents(
                        item["input_ids_q_cased"][1:-1], item["input_ids_c"][1:-1])
                    else:
                        item["input_ids"], item["segment_ids"] = self._join_sents(item["input_ids_q"][1:-1], item["input_ids_c"][1:-1])
                    item["para_offset"] = para_offset
                    item["paragraph_mask"] = torch.zeros(item["input_ids"].shape).bool()
                    item["paragraph_mask"][para_offset:-1] = 1

                    starts, ends, covered = item["starts"], item["ends"], item["covered"]
                    start_positions, end_positions = [], []

                    covered = item["covered"]
                    if covered:
                        covered = 0
                        for s, e in zip(starts, ends):
                            assert s <= e
                            if s >= paragraph.size(0):
                                continue
                            else:
                                start_position = min(
                                    s, paragraph.size(0) - 1) + para_offset
                                end_position = min(e, paragraph.size(0) - 1) + para_offset
                                covered = 1
                                start_positions.append(start_position)
                                end_positions.append(end_position)
                    if len(start_positions) == 0:
                        assert not covered
                        start_positions.append(-1)
                        end_positions.append(-1)

                    start_tensor, end_tensor, covered = torch.LongTensor(
                        start_positions), torch.LongTensor(end_positions), torch.LongTensor([covered])

                    item["start"] = start_tensor
                    item["end"] = end_tensor
                    item["covered"] = covered
                
            
                yield self.collate(batched_examples, para_embeds, topk5000_labels)
            else:
                yield {}

    def eval_load(self, retriever, k=5):
        for qa in self.qa_data:
            with torch.no_grad():
                q_ids = torch.LongTensor(self.tokenizer.encode(qa["question"], max_length=self.max_query_length)).view(1, -1).cuda()
                q_masks = torch.ones(q_ids.shape).bool().view(1, -1).cuda()
                q_cls = retriever.bert_q(q_ids, q_masks)[1]
                q_embed = retriever.proj_q(
                    q_cls).data.cpu().numpy().astype('float32')
            _, I = self.index.search(q_embed, k)
            para_embed_idx = I.reshape(-1)
            para_idx = [self.index2paraid[str(_)] for _ in para_embed_idx]
            paras = [normalize(self.para_db.get_doc_text(idx))
                     for idx in para_idx]
            para_embeds = self.para_embed[para_embed_idx]

            if self.cased:
                q_ids_cased = torch.LongTensor(self.cased_tokenizer.encode(
                    qa["question"], max_length=self.max_query_length)).view(1, -1)

            batched_examples = []
            # match answer spans in top5 paras
            for p in paras:
                p = normalize(p)

                tokenizer = self.cased_tokenizer if self.cased else self.tokenizer
                doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, all_doc_tokens = prepare(
                    p, tokenizer)

                batched_examples.append({
                    "qid": hash_question(qa["question"]),
                    "q": qa["question"],
                    "true_answers": qa["answer"],
                    "doc_toks": doc_tokens,
                    "doc_subtoks": all_doc_tokens,
                    "tok_to_orig_index": tok_to_orig_index,
                })

            for item in batched_examples:
                item["input_ids_q"] = q_ids.view(-1).cpu()

                if self.cased:
                    item["input_ids_q_cased"] = q_ids_cased.view(-1)
                    para_offset = item["input_ids_q_cased"].size(0)
                else:
                    para_offset = item["input_ids_q"].size(0)
                max_toks_for_doc = self.max_length - para_offset - 1
                para_subtoks = item["doc_subtoks"]
                if len(para_subtoks) > max_toks_for_doc:
                    para_subtoks = para_subtoks[:max_toks_for_doc]
                if self.cased:
                    p_ids = self.cased_tokenizer.convert_tokens_to_ids(
                        para_subtoks)
                else:
                    p_ids = self.tokenizer.convert_tokens_to_ids(
                        para_subtoks)
                item["input_ids_c"] = self._add_special_token(
                    torch.LongTensor(p_ids))
                paragraph = item["input_ids_c"][1:-1]
                if self.cased:
                    item["input_ids"], item["segment_ids"] = self._join_sents(
                        item["input_ids_q_cased"][1:-1], item["input_ids_c"][1:-1])
                else:
                    item["input_ids"], item["segment_ids"] = self._join_sents(
                        item["input_ids_q"][1:-1], item["input_ids_c"][1:-1])
                item["para_offset"] = para_offset
                item["paragraph_mask"] = torch.zeros(
                    item["input_ids"].shape).bool()
                item["paragraph_mask"][para_offset:-1] = 1

            yield self.collate(batched_examples, para_embeds)


    def _add_special_token(self, sent):
        cls = sent.new_full((1,), self.tokenizer.vocab["[CLS]"])
        sep = sent.new_full((1,), self.tokenizer.vocab["[SEP]"])
        sent = torch.cat([cls, sent, sep])
        return sent

    def _join_sents(self, sent1, sent2):
        cls = sent1.new_full((1,), self.tokenizer.vocab["[CLS]"])
        sep = sent1.new_full((1,), self.tokenizer.vocab["[SEP]"])
        sent1 = torch.cat([cls, sent1, sep])
        sent2 = torch.cat([sent2, sep])
        text = torch.cat([sent1, sent2])
        segment1 = torch.zeros(sent1.size(0)).long()
        segment2 = torch.ones(sent2.size(0)).long()
        segment = torch.cat([segment1, segment2])
        return text, segment

    def collate(self, samples, para_embeds, topk5000_labels=None):
        if len(samples) == 0:
            return {}

        input_ids = collate_tokens([s['input_ids'] for s in samples], 0)

        if "start" in samples[0]:
            assert topk5000_labels is not None
            net_input = {
                'input_ids': input_ids,
                'segment_ids': collate_tokens(
                    [s['segment_ids'] for s in samples], 0),
                'paragraph_mask': collate_tokens(
                    [s['paragraph_mask'] for s in samples], 0,),
                'start_positions': collate_tokens(
                    [s['start'] for s in samples], -1),
                'end_positions': collate_tokens(
                    [s['end'] for s in samples], -1),
                'para_targets': collate_tokens(
                    [s['covered'] for s in samples], 0),
                'input_mask': collate_tokens([torch.ones_like(s["input_ids"]) for s in samples], 0),
                'input_ids_q': collate_tokens([s['input_ids_q'] for s in samples], 0),
                'input_mask_q': collate_tokens([torch.ones_like(s["input_ids_q"]) for s in samples], 0),
                'para_embed': torch.from_numpy(para_embeds),
                "top5000_labels": torch.LongTensor(topk5000_labels)
            }
            return {
                'id': [s['qid'] for s in samples],
                "q": [s['q'] for s in samples],
                'wp_tokens': [s['doc_subtoks'] for s in samples],
                'para_offset': [s['para_offset'] for s in samples],
                "true_answers": [s['true_answers'] for s in samples],
                'net_input': net_input,
            }

        else:
            net_input = {
                'input_ids': input_ids,
                'segment_ids': collate_tokens(
                    [s['segment_ids'] for s in samples], 0),
                'paragraph_mask': collate_tokens(
                    [s['paragraph_mask'] for s in samples], 0,),
                'input_mask': collate_tokens([torch.ones_like(s["input_ids"]) for s in samples], 0),
                'input_ids_q': collate_tokens([s['input_ids_q'] for s in samples], 0),
                'input_mask_q': collate_tokens([torch.ones_like(s["input_ids_q"]) for s in samples], 0),
                'para_embed': torch.from_numpy(para_embeds)
            }

            return {
                'id': [s['qid'] for s in samples],
                "q": [s['q'] for s in samples],
                'doc_tokens': [s['doc_toks'] for s in samples],
                'wp_tokens': [s['doc_subtoks'] for s in samples],
                'tok_to_orig_index': [s['tok_to_orig_index'] for s in samples],
                'para_offset': [s['para_offset'] for s in samples],
                "true_answers": [s['true_answers'] for s in samples],
                'net_input': net_input,
            }


                        
if __name__ == "__main__":
    index_path = "retrieval/index_data/para_embed_3_28_c10000.npy"
    raw_data = "/home/xwhan/code/DrQA/data/datasets/nq-train.txt"


    from transformers import BertConfig, BertTokenizer
    from retrieval.retriever import BertForRetriever
    from config import get_args
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    args = get_args()
    retriever = BertForRetriever(bert_config, args)

    from utils import load_saved
    retriever_path = "retrieval/logs/splits_3_28_c10000-seed42-bsz640-fp16True-retrieve-from94_c1000_continue_from_failed-lr1e-05-bert-base-uncased-filterTrue/checkpoint_best.pt"
    retriever = load_saved(retriever, retriever_path)
    retriever.cuda()
    
    sampler = OnlineSampler(index_path, raw_data, tokenizer, args.max_query_length, args.max_seq_length)

    sampler.shuffle()
    retriever.eval()
    for batch in sampler.load(retriever):
        if batch is not {}:
            print(batch.keys())
            print(batch["net_input"]["para_targets"])
            import pdb; pdb.set_trace() 

