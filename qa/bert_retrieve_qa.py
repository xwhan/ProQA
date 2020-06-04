from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
import torch.nn.functional as F

import sys
sys.path.append('../retrieval')
from retriever import BertForRetriever


class BertRetrieveQA(nn.Module):

    def __init__(self,
        config,
        args
        ):
        super(BertRetrieveQA, self).__init__()
        self.shared_norm = args.shared_norm
        self.separate = args.separate
        self.add_select = args.add_select
        self.drop_early = args.drop_early

        if args.use_spanbert:
            self.bert = BertModel.from_pretrained(args.spanbert_path)
        else:
            self.bert = BertModel.from_pretrained(args.bert_model_name)

        # parameters from pretrained index
        self.retriever = BertForRetriever(config, args)
        if args.retriever_path != "":
            self.load_pretrained_retriever(args.retriever_path)

        self.qa_outputs = nn.Linear(
            config.hidden_size, 2)
        self.qa_drop = nn.Dropout(args.qa_drop)
        self.shared_norm = args.shared_norm

        if self.add_select:
            self.select_outputs = nn.Linear(config.hidden_size, 1)

    def load_pretrained_retriever(self, path):
        state_dict = torch.load(path)
        def filter(x): return x[7:] if x.startswith('module.') else x
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
        self.retriever.load_state_dict(state_dict)

    def freeze_c_encoder(self):
        for p in self.retriever.bert_c.parameters():
            p.requires_grad = False
        for p in self.retriever.proj_c.parameters():
            p.requires_grad = False

    def freeze_retriever(self):
        for p in self.retriever.parameters():
            p.requires_grad = False

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids = batch[
            "input_ids"], batch["input_mask"], batch["segment_ids"]
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs[0]

        logits = self.qa_drop(self.qa_outputs(sequence_output))
        outs = [o.squeeze(-1) for o in logits.split(1, dim=-1)]
        outs = [o.float().masked_fill(batch["paragraph_mask"].ne(1), -1e10).type_as(o)
                for o in outs]

        start_logits = outs[0]
        end_logits = outs[1]

        input_ids_q, attention_mask_q = batch["input_ids_q"], batch["input_mask_q"]
        q_cls = self.retriever.bert_q(input_ids_q, attention_mask_q)[1]
        q = self.retriever.proj_q(q_cls)

        rank_logits = q[0].unsqueeze(0).mm(batch["para_embed"].t())
        rank_probs = F.softmax(rank_logits, dim=-1)

        if self.add_select:
            pooled_output = outputs[1]
            select_logits = self.select_outputs(pooled_output)

        if self.training:
            start_positions, end_positions, rank_targets = batch[
                "start_positions"], batch["end_positions"], batch["para_targets"]
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")

            if not self.drop_early:
                # early loss
                para_targets = batch["top5000_labels"].nonzero()
                early_losses = [loss_fct(rank_logits, p)
                                for p in torch.unbind(para_targets)]
                if len(early_losses) == 0:
                    early_loss = loss_fct(start_logits, start_logits.new_zeros(
                        start_logits.size(0)).long()-1).sum()
                else:
                    early_loss = - \
                        torch.log(torch.sum(torch.exp(-torch.cat(early_losses))))

            if self.add_select:
                select_logits_flat = select_logits.view(1, -1)
                select_probs = F.softmax(select_logits_flat, dim=-1)

                if self.separate:
                    select_targets_flat = rank_targets.view(1, -1)
                    select_targets_flat = select_targets_flat.nonzero()[
                        :, 1].unsqueeze(1)
                    select_losses = [loss_fct(select_logits_flat, r)
                                   for r in torch.unbind(select_targets_flat)]
                    if len(select_losses) == 0:
                        select_loss = loss_fct(
                            select_logits_flat, select_logits_flat.new_zeros(1).long()-1).sum()
                    else:
                        select_loss = - torch.log(torch.sum(torch.exp(-torch.cat(select_losses))))


            # two ways to calculate the span probabilities
            if self.shared_norm:
                offset = (torch.arange(start_positions.size(
                    0)) * start_logits.size(1)).unsqueeze(1).to(start_positions.device)
                start_positions_ = start_positions + \
                    (start_positions != -1) * offset
                end_positions_ = end_positions + (end_positions != -1) * offset
                start_positions_ = start_positions_.view(-1, 1)
                end_positions_ = end_positions_.view(-1, 1)
                start_logits_flat = start_logits.view(1, -1)
                end_logits_flat = end_logits.view(1, -1)
                start_losses = [loss_fct(start_logits_flat, s)
                                for s in torch.unbind(start_positions_)]
                end_losses = [loss_fct(end_logits_flat, e)
                              for e in torch.unbind(end_positions_)]
                loss_tensor = - (torch.cat(start_losses) +
                                 torch.cat(end_losses))
                loss_tensor = loss_tensor.view(start_positions.size())
                log_prob = loss_tensor.float().masked_fill(
                    loss_tensor == 0, float('-inf')).type_as(loss_tensor)
            else:
                start_losses = [loss_fct(start_logits, starts) for starts in torch.unbind(start_positions, dim=1)]
                end_losses = [loss_fct(end_logits, ends) for ends in torch.unbind(end_positions, dim=1)]
                loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
                log_prob = - loss_tensor
                log_prob = log_prob.float().masked_fill(log_prob == 0, float('-inf')).type_as(log_prob)

            # marginal probabily for each paragraph
            probs = torch.exp(log_prob)
            marginal_probs = torch.sum(probs, dim=1)

            # joint or separate loss functions
            if self.separate:
                m_prob = [marginal_probs[idx] for idx in marginal_probs.nonzero()]
                if len(m_prob) == 0:
                    span_loss = loss_fct(start_logits, start_logits.new_zeros(
                    start_logits.size(0)).long()-1).sum()
                else:
                    span_loss = - torch.log(torch.sum(torch.cat(m_prob)))
                total_loss = span_loss + select_loss + early_loss if self.add_select else span_loss + early_loss

            else:
                if self.add_select:
                    rank_probs = select_probs

                joint_prob = marginal_probs * rank_probs.view(-1)[:marginal_probs.size(0)]
                joint_prob = [joint_prob[idx] for idx in marginal_probs.nonzero()]
                if len(joint_prob) == 0:
                    joint_loss = loss_fct(start_logits, start_logits.new_zeros(
                        start_logits.size(0)).long()-1).sum()
                else:
                    joint_loss = - torch.log(torch.sum(torch.cat(joint_prob)))
                total_loss = joint_loss + early_loss

            return {"loss": total_loss}

        if self.add_select:
            return {"start_logits": start_logits, "end_logits": end_logits, "rank_logits": rank_logits, "select_logits": select_logits.view(1, -1)}
        else:
            return {"start_logits": start_logits, "end_logits": end_logits, "rank_logits": rank_logits}
