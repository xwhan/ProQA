from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import json
import numpy as np
import random
from tqdm import tqdm
import re
import string
import os
from multiprocessing import Pool as ProcessPool

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

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


class ClusterDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_folder,
                 max_query_length,
                 max_length,
                 filter=False
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.filter = filter
        self.max_query_length = max_query_length
        self.max_length = max_length

        print(f"Loading data splits from {data_folder}")
        file_lists = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

        self.data, self.index_clusters = [], []
        processes = ProcessPool(processes=30)
        file_datas = processes.map(self.load_file, file_lists)
        processes.close()
        processes.join()
        for file_data in file_datas:
            indice = len(self.data) + np.arange(len(file_data))
            self.index_clusters.append(list(indice))
            self.data.extend(file_data)

        print(f"Total {len(self.data)} loaded")

    def filter_sample(self, item):
        if len(item["Paragraph"].split()) < 20:
            return False
        if normalize_answer(item["Answer"]) in normalize_answer(item["Question"]):
            return False
        return True

    def load_file(self, file):
        data = [json.loads(line) for line in open(file).readlines()]
        if self.filter:
            data = [item for item in data if self.filter_sample(item)]
        return data

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['Question']
        paragraph = sample['Paragraph']

        question_ids = torch.LongTensor(self.tokenizer.encode(
            question, max_length=self.max_query_length))
        question_masks = torch.ones(question_ids.shape).bool()

        paragraph_ids = torch.LongTensor(self.tokenizer.encode(
            paragraph, max_length=self.max_length - self.max_query_length))
        paragraph_masks = torch.ones(paragraph_ids.shape).bool()

        return {
            'input_ids_q': question_ids,
            'input_mask_q': question_masks,
            'input_ids_c': paragraph_ids,
            'input_mask_c': paragraph_masks,
        }

    def __len__(self):
        return len(self.data)


class ClusterSampler(Sampler):

    def __init__(self, data_source, batch_size):
        """
        batch size: within batch, all samples come from the same cluster
        """
        print(f"Sample with batch size {batch_size}")

        index_clusters = data_source.index_clusters
        sample_indice = []

        # shuffle inside each cluster
        num_group = 3
        for cluster in index_clusters:
            groups = [] # 3 adjacent examples share the same para
            for i in range(num_group):
                groups.append(cluster[i::num_group])
            random.shuffle(groups)
            for g in groups:
                random.shuffle(g)
                sample_indice += g

        # sample batches, avoid adjacent batches always come from the same cluster
        self.sample_indice = []
        batch_starts = np.arange(0, len(data_source), batch_size)
        np.random.shuffle(batch_starts)
        for batch_start in batch_starts:
            self.sample_indice += sample_indice[batch_start:batch_start+batch_size]

        assert len(self.sample_indice) == len(data_source)

    def __len__(self):
        return len(self.sample_indice)

    def __iter__(self):
        return iter(self.sample_indice)


class ReDataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_query_length,
        max_length,
        filter=False
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.filter = filter
        print(f"Loading data from {data_path}")

        self.data = [json.loads(line) for line in open(data_path).readlines()]

        # filter
        original_count = len(self.data)
        if self.filter:
            self.data = [item for item in self.data if self.filter_sample(item)]
            print(f"Using {len(self.data)} out of {original_count}")

        self.max_query_length = max_query_length
        self.max_length = max_length
        self.group_indexs = []
        num_group = 3
        indexs = list(range(len(self.data)))
        for i in range(num_group):
            self.group_indexs.append(indexs[i::num_group])

    def filter_sample(self, item):
        if len(item["Paragraph"].split()) < 20:
            return False
        if normalize_answer(item["Answer"]) in normalize_answer(item["Question"]):
            return False
        return True

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['Question']
        paragraph = sample['Paragraph']

        question_ids = torch.LongTensor(self.tokenizer.encode(question, max_length=self.max_query_length))
        question_masks = torch.ones(question_ids.shape).bool()

        paragraph_ids = torch.LongTensor(self.tokenizer.encode(paragraph, max_length=self.max_length - self.max_query_length))
        paragraph_masks = torch.ones(paragraph_ids.shape).bool()

        return {
                'input_ids_q': question_ids,
                'input_mask_q': question_masks,
                'input_ids_c': paragraph_ids,
                'input_mask_c': paragraph_masks,
                }

    def __len__(self):
        return len(self.data)


class ReSampler(Sampler):
    """
    Shuffle QA pairs not context, make sure data within the batch are from the same QA pair
    """

    def __init__(self, data_source):
        # for each QA pair, sample negative paragraphs
        sample_indice = []
        for _ in data_source.group_indexs:
            random.shuffle(_)
            sample_indice += _
        self.sample_indice = sample_indice

    def __len__(self):
        return len(self.sample_indice)

    def __iter__(self):
        return iter(self.sample_indice)

def re_collate(samples):
    if len(samples) == 0:
        return {}

    return {
            'input_ids_q': collate_tokens([s['input_ids_q'] for s in samples], 0),
            'input_mask_q': collate_tokens([s['input_mask_q'] for s in samples], 0),
            'input_ids_c': collate_tokens([s['input_ids_c'] for s in samples], 0),
            'input_mask_c': collate_tokens([s['input_mask_c'] for s in samples], 0),
        }

class FTDataset(Dataset):
    """
    finetune the Question encoder with 
    """

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_query_length,
                 max_length,
                 filter=False
                 ):
        super().__init__()


class EmDataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_query_length,
        max_length,
        is_query_embed
        ):
        super().__init__()
        self.is_query_embed = is_query_embed
        self.tokenizer = tokenizer
        print(f"Loading data from {data_path}")
        self.data = [json.loads(_.strip()) for _ in open(data_path).readlines()]

        self.max_query_length = max_query_length
        self.max_length = max_length
        self.group_indexs = []
        num_group = 3
        indexs = list(range(len(self.data)))
        for i in range(num_group):
            self.group_indexs.append(indexs[i::num_group])

    def __getitem__(self, index):
        sample = self.data[index]
        if self.is_query_embed:
            sent = sample['question']
        else:
            sent = sample['text'] if "text" in sample else sample['Paragraph']

        sent_ids = torch.LongTensor(self.tokenizer.encode(sent, max_length=self.max_query_length))
        sent_masks = torch.ones(sent_ids.shape).bool()

        return {
                'input_ids': sent_ids,
                'input_mask': sent_masks,
                }

    def __len__(self):
        return len(self.data)

def em_collate(samples):
    if len(samples) == 0:
        return {}

    return {
            'input_ids': collate_tokens([s['input_ids'] for s in samples], 0),
            'input_mask': collate_tokens([s['input_mask'] for s in samples], 0),
        }
