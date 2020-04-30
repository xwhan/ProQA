
from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch


class BertForRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super(BertForRetriever, self).__init__()

        self.bert_q = BertModel.from_pretrained(args.bert_model_name)
        self.bert_c = BertModel.from_pretrained(args.bert_model_name)

        self.proj_q = nn.Linear(config.hidden_size, 128)
        self.proj_c = nn.Linear(config.hidden_size, 128)

    def forward(self, batch):
        input_ids_q, attention_mask_q = batch["input_ids_q"], batch["input_mask_q"]
        q_cls = self.bert_q(input_ids_q, attention_mask_q)[1]
        q = self.proj_q(q_cls)

        input_ids_c, attention_mask_c = batch["input_ids_c"], batch["input_mask_c"]
        c_cls = self.bert_c(input_ids_c, attention_mask_c)[1]
        c = self.proj_c(c_cls)

        return {"q": q, "c": c}

    def get_embed(self, batch, is_query_embed):

        input_ids, attention_mask = batch["input_ids"], batch["input_mask"]
        if is_query_embed:
            q_cls = self.bert_q(input_ids, attention_mask)[1]
            q = self.proj_q(q_cls)
            return {'embed': q}
        else:
            c_cls = self.bert_c(input_ids, attention_mask)[1]
            c = self.proj_c(c_cls)
            return {'embed': c}
