from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import json
import numpy as np
import random
from tqdm import tqdm

from joblib import Parallel, delayed

from prepro_utils import hash_question

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


class OpenQASampler(Sampler):
    """
    Shuffle QA pairs not context, make sure data within the batch are from the same QA pair
    """

    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        # for each QA pair, sample negative paragraphs
        sample_indice = []
        for qa_idx in range(len(data_source.qids)):
            batch_data = []
            batch_data.append(random.choice(data_source.grouped_idx_has_answer[qa_idx]))
            assert len(batch_data) >= 1
            if len(data_source.grouped_idx_no_answer[qa_idx]) < self.batch_size - len(batch_data):
                # print("Too few negative samples...")
                # continue
                if len(data_source.grouped_idx_no_answer[qa_idx]) == 0:
                    continue
                negative_sample = random.choices(data_source.grouped_idx_no_answer[qa_idx], k=self.batch_size - len(batch_data))
            else:
                negative_sample = random.sample(data_source.grouped_idx_no_answer[qa_idx], self.batch_size - len(batch_data))
            batch_data.extend(negative_sample)
            assert len(batch_data) == batch_size
            sample_indice.append(batch_data)

        print(f"{len(sample_indice)} QA pairs used for training...")

        sample_indice = np.array(sample_indice)
        np.random.shuffle(sample_indice)
        self.sample_indice = list(sample_indice.flatten())

    def __len__(self):
        return len(self.sample_indice)

    def __iter__(self):
        return iter(self.sample_indice)


class BatchSampler(Sampler):
    """
    use all paragraphs, shuffle the QA pairs
    """

    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        sample_indice = []
        for qa_idx in range(len(data_source.qids)):
            batch_data = []
            batch_data.extend(data_source.grouped_idx_has_answer[qa_idx])
            batch_data.extend(data_source.grouped_idx_no_answer[qa_idx])
            assert len(batch_data) == batch_size
            sample_indice.append(batch_data)

        print(f"{len(sample_indice)} QA pairs used for training...")
        sample_indice = np.array(sample_indice)
        np.random.shuffle(sample_indice)
        self.sample_indice = list(sample_indice.flatten())

    def __len__(self):
        return len(self.sample_indice)

    def __iter__(self):
        return iter(self.sample_indice)
    

class OpenQADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_query_length,
                 max_length
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        print(f"Loading tokenized data from {data_path}...")
        
        
        self.qids = []
        self.all_data = [json.loads(line)
                         for line in tqdm(open(data_path).readlines())]
        self.grouped_idx_has_answer = []
        self.grouped_idx_no_answer = []
        for idx, item in enumerate(self.all_data):
            if len(self.qids) == 0 or item["qid"] != self.qids[-1]:
                self.qids.append(item["qid"])
                self.grouped_idx_no_answer.append([])
                self.grouped_idx_has_answer.append([])
            if item["no_answer"] == 0:
                self.grouped_idx_has_answer[-1].append(idx)
            else:
                self.grouped_idx_no_answer[-1].append(idx)

        print(f"{len(self.qids)} QA pairs loaded....")
        self.max_query_length = max_query_length
        self.max_length = max_length

    def __getitem__(self, index):
        sample = self.all_data[index]
        qid = sample['qid']
        q_subtoks = sample['q_subtoks']
        if len(q_subtoks) > self.max_query_length:
            q_subtoks = q_subtoks[:self.max_query_length]
        question = torch.LongTensor(self.binarize_list(q_subtoks))
        para_offset = question.size(0) + 2

        para_subtoks = sample['doc_subtoks']
        max_tokens_for_doc = self.max_length - para_offset - 1
        if len(para_subtoks) > max_tokens_for_doc:
            para_subtoks = para_subtoks[:max_tokens_for_doc]

        paragraph = torch.LongTensor(self.binarize_list(para_subtoks))
        text, seg = self._join_sents(question, paragraph)
        paragraph_mask = torch.zeros(text.shape).bool()
        question_mask = torch.zeros(text.shape).bool()
        paragraph_mask[para_offset:-1] = 1
        question_mask[1:para_offset] = 1

        starts, ends, no_answer = sample["starts"], sample["ends"], sample["no_answer"]

        start_positions, end_positions = [], []
        if not no_answer:
            no_answer = 1
            for s, e in zip(starts, ends):
                assert s <= e
                if s >= paragraph.size(0):
                    continue
                else:
                    start_position = min(s, paragraph.size(0) - 1) + para_offset
                    end_position = min(e, paragraph.size(0) - 1) + para_offset
                    no_answer = 0
                    start_positions.append(start_position)
                    end_positions.append(end_position)

        if len(start_positions) == 0:
            assert no_answer
            start_positions.append(-1)
            end_positions.append(-1)

        start_tensor, end_tensor, no_answer = torch.LongTensor(
            start_positions), torch.LongTensor(end_positions), torch.LongTensor([no_answer])

        item_tensor = {
            'q': sample["q"],
            'qid': qid,
            'input_ids': text,
            'segment_ids': seg,
            "input_ids_q": self._add_special_token(question),
            "input_ids_c": self._add_special_token(paragraph),
            'para_offset': para_offset,
            'paragraph_mask': paragraph_mask,
            'question_mask': question_mask,
            'doc_tokens': sample['doc_toks'],
            'q_subtoks': q_subtoks,
            'wp_tokens': para_subtoks,
            'tok_to_orig_index': sample['tok_to_orig_index'],
            'true_answers': sample["true_answers"],
            "start": start_tensor,
            "end": end_tensor,
            "no_answer": no_answer,
        }

        return item_tensor

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

    def _add_special_token(self, sent):
        cls = sent.new_full((1,), self.tokenizer.vocab["[CLS]"])
        sep = sent.new_full((1,), self.tokenizer.vocab["[SEP]"])
        sent = torch.cat([cls, sent, sep])
        return sent


    def binarize_list(self, words):
        return self.tokenizer.convert_tokens_to_ids(words)

    def tokenize(self, s):
        try:
            return self.tokenizer.tokenize(s)
        except:
            print('failed on', s)
            raise

    def __len__(self):
        return len(self.all_data)

def openqa_collate(samples):
    if len(samples) == 0:
        return {}

    input_ids = collate_tokens([s['input_ids'] for s in samples], 0)
    start_masks = torch.zeros(input_ids.size())
    for b_idx, s in enumerate(samples):
        for _ in s["start"]:
            if _ != -1:
                start_masks[b_idx, _] = 1
    
    net_input = {
        'input_ids': input_ids,
        'segment_ids': collate_tokens(
            [s['segment_ids'] for s in samples], 0),
        'paragraph_mask': collate_tokens(
            [s['paragraph_mask'] for s in samples], 0,),
        'question_mask': collate_tokens([s["question_mask"] for s in samples], 0),
        'start_positions': collate_tokens(
            [s['start'] for s in samples], -1),
        'end_positions': collate_tokens(
            [s['end'] for s in samples], -1),
        'no_ans_targets': collate_tokens(
            [s['no_answer'] for s in samples], 0),
        'input_mask': collate_tokens([torch.ones_like(s["input_ids"]) for s in samples], 0),
        'start_masks': start_masks,
        'input_ids_q': collate_tokens([s['input_ids_q'] for s in samples], 0),
        'input_mask_q': collate_tokens([torch.ones_like(s["input_ids_q"]) for s in samples], 0),
        'input_ids_c': collate_tokens([s['input_ids_c'] for s in samples], 0),
        'input_mask_c': collate_tokens([torch.ones_like(s["input_ids_c"]) for s in samples], 0),
    }

    return {
        'id': [s['qid'] for s in samples],
        "q": [s['q'] for s in samples],
        'doc_tokens': [s['doc_tokens'] for s in samples],
        'q_subtoks': [s['q_subtoks'] for s in samples],
        'wp_tokens': [s['wp_tokens'] for s in samples],
        'tok_to_orig_index': [s['tok_to_orig_index'] for s in samples],
        'para_offset': [s['para_offset'] for s in samples],
        "true_answers": [s['true_answers'] for s in samples],
        'net_input': net_input,
    }


class top5k_generator(object):

    def __init__(self,
                 retrieved_path,
                 embed_path
                 ):
        super().__init__()
        retrieved = [json.loads(l) for l in open(retrieved_path).readlines()]
        self.para_embed = np.load(embed_path)
        
        self.qid2para = {}
        for item in retrieved:
            self.qid2para[hash_question(item["question"])] = {"para_embed_idx": item["para_embed_idx"], "para_labels": item["para_labels"]}
    
    def generate(self, qid):
        para_labels = self.qid2para[qid]["para_labels"]
        para_embed_idx = self.qid2para[qid]["para_embed_idx"]
        if np.sum(para_labels) > 0:
            para_embed = torch.from_numpy(self.para_embed[para_embed_idx])
            para_labels = torch.tensor(para_labels).nonzero().view(-1)
            result = {}
            result["para_embed"] = para_embed
            result["para_labels"] = para_labels
            return result
        else:
            return None


if __name__ == "__main__":
    data_path = "/data/xwhan/data/mrqa-train/SQuAD-tokenized.jsonl"
    tokenized_data = [json.loads(_.strip())
                      for _ in open(data_path).readlines()]
    q_lens = np.array([len(item['q_subtoks']) for item in tokenized_data])
    c_lens = np.array([len(item['doc_subtoks']) for item in tokenized_data])
    import pdb; pdb.set_trace()
